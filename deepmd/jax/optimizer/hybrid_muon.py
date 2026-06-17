# SPDX-License-Identifier: LGPL-3.0-or-later
"""HybridMuon optimizer for the JAX backend."""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
    Callable,
    NamedTuple,
)

import optax

from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pl_triton

    PALLAS_AVAILABLE = True
except Exception:
    pl = None
    pl_triton = None
    PALLAS_AVAILABLE = False

NS_STEPS_FAST: int = 8
NS_STEPS_POLISH: int = 2
NS_COEFF_FAST: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_COEFF_POLISH: tuple[float, float, float] = (2.0, -1.5, 0.5)
NS_EPS: float = 1e-7
ADAM_EPS: float = 1e-20
FLASH_MIN_DIM: int = 1024
FLASH_BLOCK_M: int = 64
FLASH_BLOCK_K: int = 64

_GRAM_NS_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS: tuple[
    tuple[float, float, float],
    ...,
] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
GRAM_NS_SAFETY_FACTOR: float = 1.05
POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = tuple(
    (
        a / GRAM_NS_SAFETY_FACTOR,
        b / GRAM_NS_SAFETY_FACTOR**3,
        c / GRAM_NS_SAFETY_FACTOR**5,
    )
    for a, b, c in _GRAM_NS_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS
)

MAGMA_TAU: float = 2.0
MAGMA_EMA_DECAY: float = 0.9
MAGMA_MIN_SCALE: float = 0.1
MAGMA_EPS: float = 1e-12
MAGMA_SIGMOID_MIN: float = 1.0 / (1.0 + math.exp(1.0 / MAGMA_TAU))
MAGMA_SIGMOID_MAX: float = 1.0 / (1.0 + math.exp(-1.0 / MAGMA_TAU))


class HybridMuonRoute:
    """Static optimizer route for one parameter leaf."""

    __slots__ = (
        "batch_size",
        "cols",
        "kind",
        "name",
        "rows",
    )

    def __init__(
        self,
        kind: str,
        *,
        name: str,
        batch_size: int = 0,
        rows: int = 0,
        cols: int = 0,
    ) -> None:
        self.kind = kind
        self.name = name
        self.batch_size = int(batch_size)
        self.rows = int(rows)
        self.cols = int(cols)


class HybridMuonState(NamedTuple):
    """Optimizer state for JAX HybridMuon."""

    count: jnp.ndarray
    adam_mu: Any
    adam_nu: Any
    muon_momentum: Any
    magma_score: Any


def get_adam_route(param_name: str | None) -> str:
    """Return the name-based HybridMuon route used by PyTorch DPA4."""
    if param_name is None:
        return "muon"
    param_name_lower = param_name.lower()
    name_segments = param_name_lower.split(".")
    leaf_name_idx = len(name_segments) - 1
    while leaf_name_idx > 0 and name_segments[leaf_name_idx].isdigit():
        leaf_name_idx -= 1
    leaf_name = name_segments[leaf_name_idx]
    if "bias" in leaf_name:
        return "adam"
    if leaf_name.startswith("adam_"):
        return "adam"
    if leaf_name.startswith("adamw_"):
        return "adamw"
    return "muon"


def _effective_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    effective = tuple(int(dim) for dim in shape if int(dim) != 1)
    if len(effective) == 0:
        return (1,)
    return effective


def _matrix_view_shape(
    effective_shape: tuple[int, ...],
    muon_mode: str,
) -> tuple[int, int, int] | None:
    if len(effective_shape) < 2:
        return None
    if muon_mode == "2d":
        if len(effective_shape) != 2:
            return None
        return (1, int(effective_shape[-2]), int(effective_shape[-1]))
    if muon_mode == "flat":
        rows = int(math.prod(effective_shape[:-1]))
        cols = int(effective_shape[-1])
        return (1, rows, cols)
    if muon_mode == "slice":
        if len(effective_shape) == 2:
            return (1, int(effective_shape[-2]), int(effective_shape[-1]))
        batch_size = int(math.prod(effective_shape[:-2]))
        rows = int(effective_shape[-2])
        cols = int(effective_shape[-1])
        return (batch_size, rows, cols)
    raise ValueError(f"Invalid muon_mode '{muon_mode}'. Use '2d', 'flat', or 'slice'.")


def _route_param(name: str, param: Any, muon_mode: str) -> HybridMuonRoute:
    route = get_adam_route(name)
    if route == "adam":
        return HybridMuonRoute("adam", name=name)
    if route == "adamw":
        return HybridMuonRoute("adamw", name=name)

    effective = _effective_shape(tuple(int(dim) for dim in param.shape))
    if len(effective) < 2:
        return HybridMuonRoute("adam", name=name)

    matrix_shape = _matrix_view_shape(effective, muon_mode)
    if matrix_shape is None:
        return HybridMuonRoute("adamw", name=name)

    batch_size, rows, cols = matrix_shape
    return HybridMuonRoute(
        "muon",
        name=name,
        batch_size=batch_size,
        rows=rows,
        cols=cols,
    )


def build_hybrid_muon_routes(params: Any, muon_mode: str = "slice") -> Any:
    """Build a static route tree matching a parameter pytree."""
    muon_mode = str(muon_mode).lower()
    if muon_mode not in {"2d", "flat", "slice"}:
        raise ValueError(f"Invalid muon_mode '{muon_mode}'. Use '2d', 'flat', or 'slice'.")

    def walk(value: Any, path: tuple[str, ...]) -> Any:
        if hasattr(value, "items"):
            route_map = {
                key: walk(child, (*path, str(key)))
                for key, child in value.items()
            }
            if not path and hasattr(value, "to_pure_dict"):
                return type(value)(route_map)
            return route_map
        if isinstance(value, dict):
            return {
                key: walk(child, (*path, str(key)))
                for key, child in value.items()
            }
        route = _route_param(".".join(path), value, muon_mode)
        if isinstance(value, nnx.Variable):
            return type(value)(route)
        return route

    return walk(params, ())


def _zeros_like_float32(param: Any) -> jnp.ndarray:
    return jnp.zeros_like(param, dtype=jnp.float32)


def _zeros_like_param(param: Any) -> jnp.ndarray:
    return jnp.zeros_like(param)


def _init_magma_score(param: Any, route: HybridMuonRoute) -> jnp.ndarray:
    del param
    if route.kind == "muon":
        return jnp.full((route.batch_size,), 0.5, dtype=jnp.float32)
    return jnp.zeros((), dtype=jnp.float32)


def _normalize_schedule(
    learning_rate: float | Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if callable(learning_rate):
        return learning_rate
    value = float(learning_rate)
    return lambda _count: value


def _pallas_flash_available() -> bool:
    if not PALLAS_AVAILABLE:
        return False
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _pallas_matmul_transpose(x: jnp.ndarray) -> jnp.ndarray:
    """Compute ``x @ x.T`` with a Pallas/Triton symmetric matmul kernel."""
    if pl is None or pl_triton is None:
        return x @ jnp.swapaxes(x, -2, -1)

    rows, cols = (int(x.shape[0]), int(x.shape[1]))
    block_m = FLASH_BLOCK_M
    block_k = FLASH_BLOCK_K
    grid = (
        (rows + block_m - 1) // block_m,
        (rows + block_m - 1) // block_m,
    )

    def kernel(x_ref: Any, y_ref: Any) -> None:
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)

        @pl.when(pid_m <= pid_n)
        def _compute_upper_and_mirror() -> None:
            offs_m = pid_m * block_m + jnp.arange(block_m)
            offs_n = pid_n * block_m + jnp.arange(block_m)
            acc = jnp.zeros((block_m, block_m), dtype=jnp.float32)

            for col_start in range(0, cols, block_k):
                offs_k = col_start + jnp.arange(block_k)
                a = pl.load(
                    x_ref,
                    (offs_m[:, None], offs_k[None, :]),
                    mask=(offs_m[:, None] < rows) & (offs_k[None, :] < cols),
                    other=0.0,
                )
                b = pl.load(
                    x_ref,
                    (offs_n[:, None], offs_k[None, :]),
                    mask=(offs_n[:, None] < rows) & (offs_k[None, :] < cols),
                    other=0.0,
                )
                acc += pl.dot(a, b, trans_b=True, allow_tf32=True)

            out = acc.astype(x.dtype)
            out_mask = (offs_m[:, None] < rows) & (offs_n[None, :] < rows)
            pl.store(
                y_ref,
                (offs_m[:, None], offs_n[None, :]),
                out,
                mask=out_mask,
            )

            @pl.when(pid_m < pid_n)
            def _mirror() -> None:
                mirror_mask = (offs_n[:, None] < rows) & (offs_m[None, :] < rows)
                pl.store(
                    y_ref,
                    (offs_n[:, None], offs_m[None, :]),
                    jnp.swapaxes(out, 0, 1),
                    mask=mirror_mask,
                )

    compiler_params = pl_triton.CompilerParams(num_warps=4, num_stages=3)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, rows), x.dtype),
        grid=grid,
        compiler_params=compiler_params,
        name="hybrid_muon_mmt",
    )(x)


def _matmul_transpose(x: jnp.ndarray, *, use_flash: bool) -> jnp.ndarray:
    if use_flash:
        return _pallas_matmul_transpose(x)
    return x @ jnp.swapaxes(x, -2, -1)


def _newton_schulz_standard_single(
    update: jnp.ndarray,
    *,
    use_flash: bool,
) -> jnp.ndarray:
    original_dtype = update.dtype
    x = update.astype(jnp.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -2, -1)

    norm = jnp.linalg.norm(x.astype(jnp.float32), axis=(-2, -1), keepdims=True)
    x = x / jnp.maximum(norm, NS_EPS).astype(x.dtype)

    fast_a, fast_b, fast_c = NS_COEFF_FAST
    for _ in range(NS_STEPS_FAST):
        gram = _matmul_transpose(x, use_flash=use_flash)
        gram_update = fast_b * gram + fast_c * _matmul_transpose(
            gram,
            use_flash=use_flash,
        )
        x = fast_a * x + gram_update @ x

    polish_a, polish_b, polish_c = NS_COEFF_POLISH
    for _ in range(NS_STEPS_POLISH):
        gram = _matmul_transpose(x, use_flash=use_flash)
        gram_update = polish_b * gram + polish_c * _matmul_transpose(
            gram,
            use_flash=use_flash,
        )
        x = polish_a * x + gram_update @ x

    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x.astype(original_dtype)


def _newton_schulz_standard(
    update: jnp.ndarray,
    *,
    flash_muon: bool = False,
    flash_min_dim: int = FLASH_MIN_DIM,
) -> jnp.ndarray:
    """Two-stage Newton-Schulz orthogonalization for a batch of matrices."""
    use_flash = (
        bool(flash_muon)
        and _pallas_flash_available()
        and update.ndim == 3
        and int(update.shape[0]) == 1
        and min(int(update.shape[-2]), int(update.shape[-1])) >= int(flash_min_dim)
    )
    if use_flash:
        return _newton_schulz_standard_single(update[0], use_flash=True)[None, ...]

    original_dtype = update.dtype
    x = update.astype(jnp.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -2, -1)

    norm = jnp.linalg.norm(x.astype(jnp.float32), axis=(-2, -1), keepdims=True)
    x = x / jnp.maximum(norm, NS_EPS).astype(x.dtype)

    fast_a, fast_b, fast_c = NS_COEFF_FAST
    for _ in range(NS_STEPS_FAST):
        gram = x @ jnp.swapaxes(x, -2, -1)
        gram_update = fast_b * gram + fast_c * (gram @ gram)
        x = fast_a * x + gram_update @ x

    polish_a, polish_b, polish_c = NS_COEFF_POLISH
    for _ in range(NS_STEPS_POLISH):
        gram = x @ jnp.swapaxes(x, -2, -1)
        gram_update = polish_b * gram + polish_c * (gram @ gram)
        x = polish_a * x + gram_update @ x

    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x.astype(original_dtype)


def _gram_newton_schulz(update: jnp.ndarray) -> jnp.ndarray:
    """Polar-Express Gram Newton-Schulz path used by rectangular HybridMuon."""
    original_dtype = update.dtype
    x = update.astype(jnp.float32)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -2, -1)

    norm = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
    x = (x / (norm + NS_EPS)).astype(jnp.float16)
    gram = x @ jnp.swapaxes(x, -2, -1)
    identity = jnp.eye(gram.shape[-1], dtype=x.dtype)
    identity = jnp.broadcast_to(identity, gram.shape)
    transform = None
    restart_iterations = frozenset((2,))

    for idx, (coef_a, coef_b, coef_c) in enumerate(POLAR_EXPRESS_COEFFICIENTS):
        if idx in restart_iterations and idx != 0:
            x = transform @ x
            gram = x @ jnp.swapaxes(x, -2, -1)
            transform = None

        poly = coef_b * gram + coef_c * (gram @ gram)
        if idx == 0 or idx in restart_iterations:
            transform = poly + coef_a * identity
        else:
            transform = coef_a * transform + transform @ poly

        if (
            idx < len(POLAR_EXPRESS_COEFFICIENTS) - 1
            and idx + 1 not in restart_iterations
        ):
            gram_poly = coef_a * gram + gram @ poly
            gram = coef_a * gram_poly + poly @ gram_poly

    x = transform @ x
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x.astype(original_dtype)


def _orthogonalize(
    update: jnp.ndarray,
    route: HybridMuonRoute,
    enable_gram: bool,
    flash_muon: bool,
) -> jnp.ndarray:
    if enable_gram and route.rows != route.cols:
        return _gram_newton_schulz(update)
    return _newton_schulz_standard(
        update,
        flash_muon=flash_muon and not enable_gram,
    )


def _magma_scale(
    grad: jnp.ndarray,
    momentum_buffer: jnp.ndarray,
    magma_score: jnp.ndarray,
    route: HybridMuonRoute,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    grad_view = grad.reshape(route.batch_size, route.rows * route.cols).astype(
        jnp.float32
    )
    momentum_view = momentum_buffer.reshape(
        route.batch_size,
        route.rows * route.cols,
    ).astype(jnp.float32)
    dot = jnp.sum(momentum_view * grad_view, axis=1)
    denom = jnp.maximum(
        jnp.linalg.norm(momentum_view, axis=1) * jnp.linalg.norm(grad_view, axis=1),
        MAGMA_EPS,
    )
    cosine = jnp.clip(dot / denom, -1.0, 1.0)
    raw_sigmoid = jax.nn.sigmoid(cosine / MAGMA_TAU)
    raw_score = jnp.clip(
        (raw_sigmoid - MAGMA_SIGMOID_MIN)
        / (MAGMA_SIGMOID_MAX - MAGMA_SIGMOID_MIN),
        0.0,
        1.0,
    )
    new_score = MAGMA_EMA_DECAY * magma_score + (1.0 - MAGMA_EMA_DECAY) * raw_score
    return MAGMA_MIN_SCALE + (1.0 - MAGMA_MIN_SCALE) * new_score, new_score


def _adam_leaf_update(
    grad: jnp.ndarray,
    param: jnp.ndarray,
    adam_mu: jnp.ndarray,
    adam_nu: jnp.ndarray,
    count: jnp.ndarray,
    *,
    learning_rate: jnp.ndarray,
    adam_betas: tuple[float, float],
    adam_eps: float,
    weight_decay: float,
    decoupled_decay: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    beta1, beta2 = adam_betas
    grad_fp32 = grad.astype(jnp.float32)
    new_mu = beta1 * adam_mu + (1.0 - beta1) * grad_fp32
    new_nu = beta2 * adam_nu + (1.0 - beta2) * jnp.square(grad_fp32)
    count_float = count.astype(jnp.float32)
    bias_corr1 = 1.0 - beta1**count_float
    bias_corr2 = 1.0 - beta2**count_float
    step_size = learning_rate / bias_corr1
    denom = jnp.sqrt(new_nu / bias_corr2) + adam_eps
    update = -step_size * (new_mu / denom)
    if decoupled_decay and weight_decay > 0:
        update = update - learning_rate * weight_decay * param.astype(jnp.float32)
    return update.astype(param.dtype), new_mu, new_nu


def hybrid_muon(
    *,
    learning_rate: float | Callable[[jnp.ndarray], jnp.ndarray],
    params: Any,
    momentum: float = 0.95,
    weight_decay: float = 0.001,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    adam_eps: float = ADAM_EPS,
    lr_adjust: float = 0.0,
    lr_adjust_coeff: float = 0.18,
    muon_mode: str = "slice",
    enable_gram: bool = True,
    flash_muon: bool = True,
    magma_muon: bool = True,
) -> optax.GradientTransformation:
    """Create a JAX HybridMuon optax transformation.

    ``flash_muon`` uses a JAX Pallas/Triton symmetric matmul kernel for the
    same large single-matrix, non-Gram Newton-Schulz path as PyTorch. It falls
    back to XLA matmul when Pallas/Triton is unavailable, on CPU, for small
    matrices, batched slices, or when ``enable_gram`` is true.
    """
    muon_mode = str(muon_mode).lower()
    routes = build_hybrid_muon_routes(params, muon_mode)
    learning_rate_fn = _normalize_schedule(learning_rate)
    momentum = float(momentum)
    weight_decay = float(weight_decay)
    adam_betas = (float(adam_betas[0]), float(adam_betas[1]))
    adam_eps = float(adam_eps)
    lr_adjust = float(lr_adjust)
    lr_adjust_coeff = float(lr_adjust_coeff)
    enable_gram = bool(enable_gram)
    flash_muon = bool(flash_muon)
    magma_muon = bool(magma_muon)

    def init_fn(init_params: Any) -> HybridMuonState:
        return HybridMuonState(
            count=jnp.zeros([], dtype=jnp.int32),
            adam_mu=jax.tree_util.tree_map(_zeros_like_float32, init_params),
            adam_nu=jax.tree_util.tree_map(_zeros_like_float32, init_params),
            muon_momentum=jax.tree_util.tree_map(_zeros_like_param, init_params),
            magma_score=jax.tree_util.tree_map(_init_magma_score, init_params, routes),
        )

    def update_fn(
        updates: Any,
        state: HybridMuonState,
        params: Any | None = None,
        *,
        learning_rate: jnp.ndarray | float | None = None,
    ) -> tuple[Any, HybridMuonState]:
        if params is None:
            raise ValueError("JAX HybridMuon requires current parameters.")

        count_inc = optax.safe_int32_increment(state.count)
        lr = (
            learning_rate_fn(state.count)
            if learning_rate is None
            else jnp.asarray(learning_rate)
        )
        if lr_adjust <= 0:
            adam_lr = lr
        else:
            adam_lr = lr / lr_adjust

        update_leaves, treedef = jax.tree_util.tree_flatten(updates)
        param_leaves = treedef.flatten_up_to(params)
        route_leaves = treedef.flatten_up_to(routes)
        adam_mu_leaves = treedef.flatten_up_to(state.adam_mu)
        adam_nu_leaves = treedef.flatten_up_to(state.adam_nu)
        muon_momentum_leaves = treedef.flatten_up_to(state.muon_momentum)
        magma_score_leaves = treedef.flatten_up_to(state.magma_score)

        out_updates = []
        out_adam_mu = []
        out_adam_nu = []
        out_muon_momentum = []
        out_magma_score = []

        for (
            grad,
            param,
            route,
            adam_mu,
            adam_nu,
            muon_momentum,
            magma_score,
        ) in zip(
            update_leaves,
            param_leaves,
            route_leaves,
            adam_mu_leaves,
            adam_nu_leaves,
            muon_momentum_leaves,
            magma_score_leaves,
            strict=True,
        ):
            if not jnp.issubdtype(param.dtype, jnp.inexact):
                out_updates.append(jnp.zeros_like(param))
                out_adam_mu.append(adam_mu)
                out_adam_nu.append(adam_nu)
                out_muon_momentum.append(muon_momentum)
                out_magma_score.append(magma_score)
                continue

            if route.kind in {"adam", "adamw"}:
                leaf_update, new_mu, new_nu = _adam_leaf_update(
                    grad,
                    param,
                    adam_mu,
                    adam_nu,
                    count_inc,
                    learning_rate=adam_lr,
                    adam_betas=adam_betas,
                    adam_eps=adam_eps,
                    weight_decay=weight_decay,
                    decoupled_decay=route.kind == "adamw",
                )
                out_updates.append(leaf_update)
                out_adam_mu.append(new_mu)
                out_adam_nu.append(new_nu)
                out_muon_momentum.append(muon_momentum)
                out_magma_score.append(magma_score)
                continue

            new_momentum = momentum * muon_momentum + (1.0 - momentum) * grad
            muon_update = momentum * new_momentum + (1.0 - momentum) * grad
            matrix_update = muon_update.reshape(
                route.batch_size,
                route.rows,
                route.cols,
            )
            orthogonalized = _orthogonalize(
                matrix_update,
                route,
                enable_gram,
                flash_muon,
            )
            if lr_adjust <= 0:
                scale = lr_adjust_coeff * math.sqrt(float(max(route.rows, route.cols)))
            else:
                scale = max(1.0, route.rows / route.cols) ** 0.5
            orthogonalized = orthogonalized * scale

            if magma_muon:
                scale_magma, new_magma_score = _magma_scale(
                    grad.reshape(route.batch_size, route.rows, route.cols),
                    new_momentum.reshape(route.batch_size, route.rows, route.cols),
                    magma_score,
                    route,
                )
                orthogonalized = orthogonalized * scale_magma.reshape(
                    route.batch_size,
                    1,
                    1,
                ).astype(orthogonalized.dtype)
            else:
                new_magma_score = magma_score

            leaf_update = -lr * orthogonalized.reshape(param.shape)
            if weight_decay > 0:
                leaf_update = leaf_update - lr * weight_decay * param
            out_updates.append(leaf_update.astype(param.dtype))
            out_adam_mu.append(adam_mu)
            out_adam_nu.append(adam_nu)
            out_muon_momentum.append(new_momentum)
            out_magma_score.append(new_magma_score)

        return (
            jax.tree_util.tree_unflatten(treedef, out_updates),
            HybridMuonState(
                count=count_inc,
                adam_mu=jax.tree_util.tree_unflatten(treedef, out_adam_mu),
                adam_nu=jax.tree_util.tree_unflatten(treedef, out_adam_nu),
                muon_momentum=jax.tree_util.tree_unflatten(treedef, out_muon_momentum),
                magma_score=jax.tree_util.tree_unflatten(treedef, out_magma_score),
            ),
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
