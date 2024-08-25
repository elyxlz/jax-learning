"""a rectified flow dit in pure functional jax"""

from dataclasses import dataclass
from typing import NamedTuple, Callable
from tqdm import trange
from functools import partial
import jax
import jax.numpy as jnp


@dataclass
class DiTConfig:
    in_dim: int = 16
    patch_size: int = 2
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4


def init_layernorm(dim: int) -> jax.Array:
    return jnp.ones(dim)


def layernorm(x: jax.Array, params: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    return jnp.multiply(x, params)


def init_linear(in_dim: int, out_dim: int, key: jax.typing.ArrayLike) -> jax.Array:
    k = 1 / (in_dim**0.5)
    return jax.random.uniform(key, (in_dim, out_dim), minval=-k, maxval=k)


def linear(x: jax.Array, params: jax.Array) -> jax.Array:
    return jnp.matmul(x, params)


class MlpParams(NamedTuple):
    norm: jax.Array
    fc1: jax.Array
    fc2: jax.Array
    fc3: jax.Array


def init_mlp(dim: int, inner_dim: int, key: jax.typing.ArrayLike) -> MlpParams:
    k1, k2, k3 = jax.random.split(key, num=3)
    return MlpParams(
        norm=init_layernorm(dim),
        fc1=init_linear(dim, out_dim=inner_dim, key=k1),
        fc2=init_linear(dim, out_dim=inner_dim, key=k2),
        fc3=init_linear(inner_dim, out_dim=dim, key=k3),
    )


def mlp(x: jax.Array, params: MlpParams) -> jax.Array:
    h = jax.nn.silu(linear(x, params=params.fc1))
    gate = linear(x, params=params.fc2)
    x = jnp.multiply(h, gate)
    return linear(x, params=params.fc3)


class AttentionParams(NamedTuple):
    norm: jax.Array
    qkv: jax.Array
    qnorm: jax.Array
    knorm: jax.Array
    o: jax.Array


def init_attention(
    dim: int, head_dim: int, key: jax.typing.ArrayLike
) -> AttentionParams:
    k1, k2 = jax.random.split(key, num=2)
    return AttentionParams(
        norm=init_layernorm(dim),
        qkv=init_linear(dim, out_dim=dim * 3, key=k1),
        qnorm=init_layernorm(head_dim),
        knorm=init_layernorm(head_dim),
        o=init_linear(dim, out_dim=dim, key=k2),
    )


# TODO: rope
def attention(x: jax.Array, params: AttentionParams, num_heads: int) -> jax.Array:
    s, d = x.shape
    qkv = linear(x, params=params.qkv)
    qkv = jnp.reshape(qkv, shape=(1, s, num_heads, d * 3 // num_heads))
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = layernorm(q, params=params.qnorm)
    k = layernorm(k, params=params.knorm)
    x = jnp.reshape(jax.nn.dot_product_attention(q, k, v), shape=(s, d))
    x = linear(x, params=params.o)
    return layernorm(x, params.norm)


class TransformerLayerParams(NamedTuple):
    mlp: MlpParams
    attention: AttentionParams


def init_transformer_layer(
    config: DiTConfig, key: jax.typing.ArrayLike
) -> TransformerLayerParams:
    k1, k2 = jax.random.split(key)
    inner_dim = int(128 * max(1, int(config.hidden_dim * 8 / 3) // 128))
    return TransformerLayerParams(
        mlp=init_mlp(config.hidden_dim, inner_dim=inner_dim, key=k1),
        attention=init_attention(
            config.hidden_dim, head_dim=config.hidden_dim // config.num_heads, key=k2
        ),
    )


def transformer_layer(x: jax.Array, params: TransformerLayerParams, config: DiTConfig):
    x = attention(x, params=params.attention, num_heads=config.num_heads) + x
    return mlp(x, params=params.mlp) + x


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
        layers=[init_transformer_layer(config, key=k) for k in keys[1:-1]],
        norm=init_layernorm(config.hidden_dim),
        proj_out=init_linear(
            config.hidden_dim // config.patch_size, out_dim=config.in_dim, key=keys[-1]
        ),
    )


def dit(x: jax.Array, params: DiTParams, config: DiTConfig) -> jax.Array:
    seq_len = x.shape[0]
    x = linear(x, params=params.proj_in)
    x = jnp.reshape(x, shape=(seq_len // config.patch_size, config.hidden_dim))

    for layer in params.layers:
        x = transformer_layer(x, params=layer, config=config)

    x = layernorm(x, params=params.norm)
    x = jnp.reshape(x, (seq_len, config.hidden_dim // config.patch_size))
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
    config = DiTConfig()
    dit_params = init_dit(config, key)
    out = jax.vmap(partial(dit, params=dit_params, config=config))(arr)
    print(out.shape)
