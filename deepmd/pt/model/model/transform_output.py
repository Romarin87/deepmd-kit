# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_hessian_name,
    get_reduce_name,
)
from deepmd.pt.utils import (
    env,
)


def atomic_virial_corr(
    extended_coord: torch.Tensor,
    atom_energy: torch.Tensor,
) -> torch.Tensor:
    nall = extended_coord.shape[1]
    nloc = atom_energy.shape[1]
    coord, _ = torch.split(extended_coord, [nloc, nall - nloc], dim=1)
    # no derivative with respect to the loc coord.
    coord = coord.detach()
    ce = coord * atom_energy
    sumce0, sumce1, sumce2 = torch.split(torch.sum(ce, dim=1), [1, 1, 1], dim=-1)
    faked_grad = torch.ones_like(sumce0)
    lst = torch.jit.annotate(list[torch.Tensor | None], [faked_grad])
    extended_virial_corr0 = torch.autograd.grad(
        [sumce0],
        [extended_coord],
        grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr0 is not None
    extended_virial_corr1 = torch.autograd.grad(
        [sumce1],
        [extended_coord],
        grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr1 is not None
    extended_virial_corr2 = torch.autograd.grad(
        [sumce2],
        [extended_coord],
        grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr2 is not None
    extended_virial_corr = torch.concat(
        [
            extended_virial_corr0.unsqueeze(-1),
            extended_virial_corr1.unsqueeze(-1),
            extended_virial_corr2.unsqueeze(-1),
        ],
        dim=-1,
    )
    return extended_virial_corr


def task_deriv_one(
    atom_energy: torch.Tensor,
    energy: torch.Tensor,
    extended_coord: torch.Tensor,
    do_virial: bool = True,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    faked_grad = torch.ones_like(energy)
    lst = torch.jit.annotate(list[torch.Tensor | None], [faked_grad])
    extended_force = torch.autograd.grad(
        [energy],
        [extended_coord],
        grad_outputs=lst,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    assert extended_force is not None
    extended_force = -extended_force
    if do_virial:
        extended_virial = torch.einsum(
            "...ik,...ij->...ikj", extended_force, extended_coord
        )
        # the correction sums to zero, which does not contribute to global virial
        if do_atomic_virial:
            extended_virial_corr = atomic_virial_corr(extended_coord, atom_energy)
            extended_virial = extended_virial + extended_virial_corr
        # to [...,3,3] -> [...,9]
        extended_virial = extended_virial.view(list(extended_virial.shape[:-2]) + [9])  # noqa:RUF005
    else:
        extended_virial = None
    return extended_force, extended_virial


def get_leading_dims(
    vv: torch.Tensor,
    vdef: OutputVariableDef,
) -> list[int]:
    """Get the dimensions of nf x nloc."""
    vshape = vv.shape
    return list(vshape[: (len(vshape) - len(vdef.shape))])


def take_deriv(
    vv: torch.Tensor,
    svv: torch.Tensor,
    vdef: OutputVariableDef,
    coord_ext: torch.Tensor,
    do_virial: bool = False,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    size = 1
    for ii in vdef.shape:
        size *= ii
    vv1 = vv.view(list(get_leading_dims(vv, vdef)) + [size])  # noqa: RUF005
    svv1 = svv.view(list(get_leading_dims(svv, vdef)) + [size])  # noqa: RUF005
    split_vv1 = torch.split(vv1, [1] * size, dim=-1)
    split_svv1 = torch.split(svv1, [1] * size, dim=-1)
    split_ff, split_avir = [], []
    for vvi, svvi in zip(split_vv1, split_svv1):
        # nf x nloc x 3, nf x nloc x 9
        ffi, aviri = task_deriv_one(
            vvi,
            svvi,
            coord_ext,
            do_virial=do_virial,
            do_atomic_virial=do_atomic_virial,
            create_graph=create_graph,
        )
        # nf x nloc x 1 x 3, nf x nloc x 1 x 9
        ffi = ffi.unsqueeze(-2)
        split_ff.append(ffi)
        if do_virial:
            assert aviri is not None
            aviri = aviri.unsqueeze(-2)
            split_avir.append(aviri)
    # nf x nall x v_dim x 3, nf x nall x v_dim x 9
    out_lead_shape = list(coord_ext.shape[:-1]) + vdef.shape
    ff = torch.concat(split_ff, dim=-2).view(out_lead_shape + [3])  # noqa: RUF005
    if do_virial:
        avir = torch.concat(split_avir, dim=-2).view(out_lead_shape + [9])  # noqa: RUF005
    else:
        avir = None
    return ff, avir


def take_hessian(
    svv: torch.Tensor,
    vdef: OutputVariableDef,
    coord_ext: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    size = 1
    for ii in vdef.shape:
        size *= ii
    svv1 = svv.view(list(get_leading_dims(svv, vdef)) + [size])  # noqa: RUF005
    nf = coord_ext.shape[0]
    nall = coord_ext.shape[1]
    hessian_components = []
    for idx in range(size):
        frame_hessians = []
        for iframe in range(nf):
            energy_component = svv1[iframe, idx]
            grad = torch.autograd.grad(
                [energy_component],
                [coord_ext],
                grad_outputs=torch.jit.annotate(
                    list[torch.Tensor | None], [torch.ones_like(energy_component)]
                ),
                create_graph=True,
                retain_graph=True,
            )[0]
            assert grad is not None
            flat_grad = grad[iframe].reshape(-1)
            rows = []
            for gcomp in flat_grad:
                second = torch.autograd.grad(
                    [gcomp],
                    [coord_ext],
                    grad_outputs=torch.jit.annotate(
                        list[torch.Tensor | None], [torch.ones_like(gcomp)]
                    ),
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if second is None:
                    rows.append(torch.zeros_like(coord_ext[iframe]).reshape(-1))
                else:
                    rows.append(second[iframe].reshape(-1))
            frame_hessians.append(
                torch.stack(rows, dim=0).view(nall, 3, nall, 3)
            )
        hessian_components.append(torch.stack(frame_hessians, dim=0))
    return torch.stack(hessian_components, dim=1).view(
        [nf] + list(vdef.shape) + [nall, 3, nall, 3]
    )


def fit_output_to_model_output(
    fit_ret: dict[str, torch.Tensor],
    fit_output_def: FittingOutputDef,
    coord_ext: torch.Tensor,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
    mask: torch.Tensor | None = None,
    extended_coord_corr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Transform the output of the fitting network to
    the model output.

    """
    redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            if vdef.intensive:
                if mask is not None:
                    model_ret[kk_redu] = torch.sum(
                        vv.to(redu_prec), dim=atom_axis
                    ) / torch.sum(mask, dim=-1, keepdim=True)
                else:
                    model_ret[kk_redu] = torch.mean(vv.to(redu_prec), dim=atom_axis)
            else:
                model_ret[kk_redu] = torch.sum(vv.to(redu_prec), dim=atom_axis)
            if vdef.r_differentiable:
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                dr, dc = take_deriv(
                    vv,
                    model_ret[kk_redu],
                    vdef,
                    coord_ext,
                    do_virial=vdef.c_differentiable,
                    do_atomic_virial=do_atomic_virial,
                    create_graph=create_graph,
                )
                model_ret[kk_derv_r] = dr
                if vdef.c_differentiable:
                    assert dc is not None
                    if extended_coord_corr is not None:
                        dc_corr = (
                            dr.squeeze(-2).unsqueeze(-1)
                            @ extended_coord_corr.unsqueeze(-2).to(dr.dtype)
                        ).view(list(dc.shape[:-2]) + [1, 9])  # noqa: RUF005
                        dc = dc + dc_corr
                    model_ret[kk_derv_c] = dc
                    model_ret[kk_derv_c + "_redu"] = torch.sum(
                        model_ret[kk_derv_c].to(redu_prec), dim=1
                    )
                if vdef.r_hessian:
                    kk_hessian = get_hessian_name(kk)
                    model_ret[kk_hessian] = take_hessian(
                        model_ret[kk_redu],
                        vdef,
                        coord_ext,
                        create_graph=create_graph,
                    )
    return model_ret


def communicate_extended_output(
    model_ret: dict[str, torch.Tensor],
    model_output_def: ModelOutputDef,
    mapping: torch.Tensor,  # nf x nloc
    do_atomic_virial: bool = False,
) -> dict[str, torch.Tensor]:
    """Transform the output of the model network defined on
    local and ghost (extended) atoms to local atoms.

    """
    redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
    new_ret = {}
    for kk in model_output_def.keys_outp():
        vv = model_ret[kk]
        vdef = model_output_def[kk]
        new_ret[kk] = vv
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            new_ret[kk_redu] = model_ret[kk_redu]
            mapping_base = mapping
            # nf x nloc
            vldims = get_leading_dims(vv, vdef)
            # nf x nall
            mldims = list(mapping_base.shape)
            kk_derv_r, kk_derv_c = get_deriv_name(kk)
            if vdef.r_differentiable:
                # vdim x 3
                derv_r_ext_dims = list(vdef.shape) + [3]  # noqa:RUF005
                mapping_derv_r = mapping_base.view(
                    mldims + [1] * len(derv_r_ext_dims)
                ).expand(
                    [-1] * len(mldims) + derv_r_ext_dims
                )
                force = torch.zeros(
                    vldims + derv_r_ext_dims, dtype=vv.dtype, device=vv.device
                )
                # nf x nloc x nvar x 3
                new_ret[kk_derv_r] = torch.scatter_reduce(
                    force,
                    1,
                    index=mapping_derv_r,
                    src=model_ret[kk_derv_r],
                    reduce="sum",
                )
                if vdef.r_hessian:
                    kk_hessian = get_hessian_name(kk)
                    if model_ret.get(kk_hessian) is not None:
                        hess = model_ret[kk_hessian]
                        def_ndim = len(vdef.shape)
                        # [nf, *def, nall1, 3, nall2, 3]
                        hess_1 = hess.permute(
                            0,
                            def_ndim + 1,
                            def_ndim + 3,
                            *range(1, def_ndim + 1),
                            def_ndim + 2,
                            def_ndim + 4,
                        )
                        nall = hess_1.shape[1]
                        hessian1 = torch.zeros(
                            [*vldims, nall, *vdef.shape, 3, 3],
                            dtype=vv.dtype,
                            device=vv.device,
                        )
                        mapping_hess = mapping_base.view(
                            mldims + [1] * (len(vdef.shape) + 3)
                        ).expand(
                            [-1] * len(mldims) + [nall, *vdef.shape, 3, 3]
                        )
                        hessian1 = torch.scatter_reduce(
                            hessian1,
                            1,
                            index=mapping_hess,
                            src=hess_1,
                            reduce="sum",
                        )
                        hessian1 = hessian1.permute(
                            0, 2, 1, *range(3, def_ndim + 5)
                        )
                        nloc = hessian1.shape[2]
                        hessian = torch.zeros(
                            [*vldims, nloc, *vdef.shape, 3, 3],
                            dtype=vv.dtype,
                            device=vv.device,
                        )
                        mapping_hess = mapping_base.view(
                            mldims + [1] * (len(vdef.shape) + 3)
                        ).expand(
                            [-1] * len(mldims) + [nloc, *vdef.shape, 3, 3]
                        )
                        hessian = torch.scatter_reduce(
                            hessian,
                            1,
                            index=mapping_hess,
                            src=hessian1,
                            reduce="sum",
                        )
                        hessian = hessian.permute(
                            0,
                            *range(3, def_ndim + 3),
                            2,
                            def_ndim + 3,
                            1,
                            def_ndim + 4,
                        )
                        new_ret[kk_hessian] = hessian.reshape(
                            hessian.shape[0], *vdef.shape, nloc * 3, nloc * 3
                        )
                    else:
                        new_ret[kk_hessian] = None
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                derv_c_ext_dims = list(vdef.shape) + [9]  # noqa:RUF005
                # nf x nloc x nvar x 3 -> nf x nloc x nvar x 9
                mapping_derv_c = torch.tile(
                    mapping_derv_r,
                    [1] * (len(mldims) + len(vdef.shape)) + [3],
                )
                virial = torch.zeros(
                    vldims + derv_c_ext_dims, dtype=vv.dtype, device=vv.device
                )
                # nf x nloc x nvar x 9
                new_ret[kk_derv_c] = torch.scatter_reduce(
                    virial,
                    1,
                    index=mapping_derv_c,
                    src=model_ret[kk_derv_c],
                    reduce="sum",
                )
                new_ret[kk_derv_c + "_redu"] = torch.sum(
                    new_ret[kk_derv_c].to(redu_prec), dim=1
                )
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
