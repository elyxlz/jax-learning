"""a rectified flow dit in pure functional jax"""

from dataclasses import dataclass
from typing import NamedTuple, Callable
from tqdm import trange
from functools import partial
import jax
import jax.numpy as jnp


def build_rope_cache(seq_len: int, n_elem: int, base: int = 10000) -> jax.Array:
    freqs = 1.0 / (
        base ** (jnp.arange(0, n_elem, 2)[: (n_elem // 2)].astype(jnp.float32) / n_elem)
    )
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return jnp.stack([jnp.real(freqs_cis), jnp.imag(freqs_cis)], axis=-1)


def apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    freqs = freqs[: x.shape[2]]
    xshaped = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    freqs = freqs.reshape(1, 1, xshaped.shape[2], xshaped.shape[3], 2)
    x_out = jnp.stack(
        [
            xshaped[..., 0] * freqs[..., 0] - xshaped[..., 1] * freqs[..., 1],
            xshaped[..., 1] * freqs[..., 0] + xshaped[..., 0] * freqs[..., 1],
        ],
        -1,
    )
    return x_out.reshape(*x.shape[:-1], -1).astype(x.dtype)


@dataclass
class DiTConfig:
    in_dim: int = 16
    patch_size: int = 2
    hidden_dim: int = 64
    time_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4


def init_layernorm(dim: int) -> jax.Array:
    return jnp.ones(dim)


def layernorm(x: jax.Array, params: jax.Array, eps: float = 1e-5) -> jax.Array:
    x_32 = x.astype(jnp.float32)
    mean = jnp.mean(x_32, axis=-1, keepdims=True)
    var = jnp.var(x_32, axis=-1, keepdims=True)
    x_32 = (x_32 - mean) / jnp.sqrt(var + eps)
    return jnp.multiply(x_32, params).astype(x.dtype)


def init_linear(
    in_dim: int, out_dim: int, key: jax.typing.ArrayLike, zero: bool = False
) -> jax.Array:
    k = 1 / (in_dim**0.5) if not zero else 0
    return jax.random.uniform(key, (in_dim, out_dim), minval=-k, maxval=k)


def linear(x: jax.Array, params: jax.Array) -> jax.Array:
    return jnp.matmul(x, params)


class MlpParams(NamedTuple):
    norm: jax.Array
    modulation: jax.Array
    fc1: jax.Array
    fc2: jax.Array
    fc3: jax.Array


def init_mlp(
    dim: int, inner_dim: int, modulation_dim: int, key: jax.typing.ArrayLike
) -> MlpParams:
    k1, k2, k3, k4 = jax.random.split(key, num=4)
    return MlpParams(
        norm=init_layernorm(dim),
        modulation=init_linear(modulation_dim, out_dim=dim * 3, key=k1, zero=True),
        fc1=init_linear(dim, out_dim=inner_dim, key=k2),
        fc2=init_linear(dim, out_dim=inner_dim, key=k3),
        fc3=init_linear(inner_dim, out_dim=dim, key=k4),
    )


def mlp(x: jax.Array, modulation: jax.Array, params: MlpParams) -> jax.Array:
    x = layernorm(x, params=params.norm)
    shift, scale, gate = jnp.split(
        linear(modulation, params=params.modulation), 3, axis=-1
    )
    x = jnp.multiply(x + shift, scale + 1)
    h = jax.nn.silu(linear(x, params=params.fc1))
    mlp_gate = linear(x, params=params.fc2)
    x = jnp.multiply(h, mlp_gate)
    return jnp.multiply(linear(x, params=params.fc3), gate)


class AttentionParams(NamedTuple):
    norm: jax.Array
    modulation: jax.Array
    qkv: jax.Array
    qnorm: jax.Array
    knorm: jax.Array
    o: jax.Array


def init_attention(
    dim: int, head_dim: int, modulation_dim: int, key: jax.typing.ArrayLike
) -> AttentionParams:
    k1, k2, k3 = jax.random.split(key, num=3)
    return AttentionParams(
        norm=init_layernorm(dim),
        modulation=init_linear(modulation_dim, out_dim=dim * 3, key=k1, zero=True),
        qkv=init_linear(dim, out_dim=dim * 3, key=k2),
        qnorm=init_layernorm(head_dim),
        knorm=init_layernorm(head_dim),
        o=init_linear(dim, out_dim=dim, key=k3),
    )


# TODO: rope
def attention(
    x: jax.Array, modulation: jax.Array, params: AttentionParams, num_heads: int
) -> jax.Array:
    s, d = x.shape
    x = layernorm(x, params=params.norm)
    shift, scale, gate = jnp.split(
        linear(modulation, params=params.modulation), 3, axis=-1
    )
    x = jnp.multiply(x + shift, scale + 1)
    qkv = linear(x, params=params.qkv)
    qkv = jnp.reshape(qkv, shape=(1, s, num_heads, d * 3 // num_heads))
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = layernorm(q, params=params.qnorm)
    k = layernorm(k, params=params.knorm)
    x = jnp.reshape(jax.nn.dot_product_attention(q, k, v), shape=(s, d))
    x = linear(x, params=params.o)
    return jnp.multiply(x, gate)


class TransformerLayerParams(NamedTuple):
    mlp: MlpParams
    attention: AttentionParams


def init_transformer_layer(
    dim: int,
    modulation_dim: int,
    num_heads: int,
    key: jax.typing.ArrayLike,
) -> TransformerLayerParams:
    k1, k2 = jax.random.split(key)
    inner_dim = int(128 * max(1, int(config.hidden_dim * 8 / 3) // 128))
    return TransformerLayerParams(
        mlp=init_mlp(
            dim,
            inner_dim=inner_dim,
            modulation_dim=modulation_dim,
            key=k1,
        ),
        attention=init_attention(
            dim,
            head_dim=dim // num_heads,
            modulation_dim=modulation_dim,
            key=k2,
        ),
    )


def transformer_layer(
    x: jax.Array,
    modulation: jax.Array,
    params: TransformerLayerParams,
    config: DiTConfig,
):
    x = (
        attention(
            x,
            modulation=modulation,
            params=params.attention,
            num_heads=config.num_heads,
        )
        + x
    )
    return mlp(x, modulation=modulation, params=params.mlp) + x


class DiTParams(NamedTuple):
    proj_in: jax.Array
    layers: list[TransformerLayerParams]
    norm: jax.Array
    proj_out: jax.Array


def init_dit(config: DiTConfig, key: jax.typing.ArrayLike) -> DiTParams:
    keys = jax.random.split(key, config.num_layers + 2)
    return DiTParams(
        proj_in=init_linear(
            config.in_dim, out_dim=config.hidden_dim // config.patch_size, key=keys[0]
        ),
        layers=[
            init_transformer_layer(
                config.hidden_dim,
                modulation_dim=config.time_dim,
                num_heads=config.num_heads,
                key=k,
            )
            for k in keys[1:-1]
        ],
        norm=init_layernorm(config.hidden_dim),
        proj_out=init_linear(
            config.hidden_dim // config.patch_size,
            out_dim=config.in_dim,
            key=keys[-1],
            zero=True,
        ),
    )


def dit(
    x: jax.Array, time: jax.Array, params: DiTParams, config: DiTConfig
) -> jax.Array:
    seq_len = x.shape[0]
    x = linear(x, params=params.proj_in)
    x = jnp.reshape(x, shape=(seq_len // config.patch_size, config.hidden_dim))

    def scan_fn(carry, layer):
        return transformer_layer(
            carry, modulation=time, params=layer, config=config
        ), None

    layers_stacked = jax.tree_util.tree_map(
        lambda *args: jnp.stack(args), *params.layers
    )  # does this cause a slow down?
    x, _ = jax.lax.scan(scan_fn, x, layers_stacked)

    x = layernorm(x, params=params.norm)
    x = jnp.reshape(x, (seq_len, -1))
    return linear(x, params=params.proj_out)


def generate(
    dit_params: DiTParams,
    bs: int,
    seq_len: int,
    steps: int,
    key: jax.typing.ArrayLike,
    config: DiTConfig,
) -> jax.Array:
    noise = jax.random.normal(key, shape(bs, seq_len, config.in_dim))
    dt = 1.0 / steps
    for t in trange(steps):
        raise NotImplementedError


if __name__ == "__main__":
    key = jax.random.key(42)
    shape = (8, 100, 16)
    arr = jax.random.normal(key, shape)
    mod = jax.random.normal(key, (8, 1, 64))
    config = DiTConfig()
    dit_params = init_dit(config, key)
    out = jax.vmap(partial(dit, params=dit_params, config=config))(arr, time=mod)
    print(out.shape)
