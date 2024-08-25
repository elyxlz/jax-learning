"""a modern 1d rectified flow dit in pure functional jax"""

import typing
from functools import partial

import jax
import jax.numpy as jnp


class DiTConfig(typing.NamedTuple):
    in_dim: int = 16
    patch_size: int = 2
    hidden_dim: int = 64
    time_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 100
    rope_base: int = 10_000


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


def init_rope(seq_len: int, num_elem: int, base: int = 10000) -> jax.Array:
    freqs = 1.0 / (
        base ** (jnp.arange(0, num_elem, 2)[: (num_elem // 2)].astype(jnp.float32) / num_elem)
    )
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return jnp.stack([jnp.real(freqs_cis), jnp.imag(freqs_cis)], axis=-1)


def rope(x: jax.Array, params: jax.Array) -> jax.Array:
    freqs = jax.lax.stop_gradient(params)[: x.shape[2]]
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


class FourierFeaturesParams(typing.NamedTuple):
    scales: jax.Array
    to_out: jax.Array


def init_fourier_features(time_dim: int, key: jax.typing.ArrayLike) -> FourierFeaturesParams:
    k1, k2 = jax.random.split(key)
    assert time_dim % 2 == 0
    scales = jax.random.normal(k1, (time_dim // 2, 1)) * 1.0
    to_out = init_linear(time_dim, time_dim, k2)
    return FourierFeaturesParams(scales=scales, to_out=to_out)


def fourier_features(x: jax.Array, params: FourierFeaturesParams) -> jax.Array:
    f = 2 * jnp.pi * jnp.matmul(x, jax.lax.stop_gradient(params.scales).T)
    fouriered = jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
    return linear(fouriered, params.to_out)


class MlpParams(typing.NamedTuple):
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
    shift, scale, gate = jnp.split(linear(modulation, params=params.modulation), 3, axis=-1)
    x = jnp.multiply(x + shift, scale + 1)
    h = jax.nn.silu(linear(x, params=params.fc1))
    mlp_gate = linear(x, params=params.fc2)
    x = jnp.multiply(h, mlp_gate)
    return jnp.multiply(linear(x, params=params.fc3), gate)


class AttentionParams(typing.NamedTuple):
    norm: jax.Array
    modulation: jax.Array
    qkv: jax.Array
    qk_norm: tuple[jax.Array, jax.Array]
    rope_cache: jax.Array
    o: jax.Array


def init_attention(
    dim: int,
    head_dim: int,
    modulation_dim: int,
    seq_len: int,
    rope_base: int,
    key: jax.typing.ArrayLike,
) -> AttentionParams:
    k1, k2, k3 = jax.random.split(key, num=3)
    return AttentionParams(
        norm=init_layernorm(dim),
        modulation=init_linear(modulation_dim, out_dim=dim * 3, key=k1, zero=True),
        qkv=init_linear(dim, out_dim=dim * 3, key=k2),
        qk_norm=(init_layernorm(head_dim), init_layernorm(head_dim)),
        rope_cache=init_rope(seq_len, num_elem=head_dim, base=rope_base),
        o=init_linear(dim, out_dim=dim, key=k3),
    )


def attention(x: jax.Array, modulation: jax.Array, params: AttentionParams) -> jax.Array:
    s, d = x.shape
    x = layernorm(x, params=params.norm)
    shift, scale, gate = jnp.split(linear(modulation, params=params.modulation), 3, axis=-1)
    x = jnp.multiply(x + shift, scale + 1)
    qkv = linear(x, params=params.qkv)
    num_heads = x.shape[-1] // params.qk_norm[0].shape[0]
    qkv = jnp.reshape(qkv, shape=(1, s, num_heads, d * 3 // num_heads))
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = layernorm(q, params=params.qk_norm[0])
    k = layernorm(k, params=params.qk_norm[1])
    q = rope(q, params.rope_cache)
    k = rope(k, params.rope_cache)
    x = jnp.reshape(jax.nn.dot_product_attention(q, k, v), shape=(s, d))
    x = linear(x, params=params.o)
    return jnp.multiply(x, gate)


class TransformerLayerParams(typing.NamedTuple):
    mlp: MlpParams
    attention: AttentionParams


def init_transformer_layer(
    dim: int,
    modulation_dim: int,
    num_heads: int,
    seq_len: int,
    rope_base: int,
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
            seq_len=seq_len,
            rope_base=rope_base,
            key=k2,
        ),
    )


def transformer_layer(
    x: jax.Array,
    modulation: jax.Array,
    params: TransformerLayerParams,
):
    x = (
        attention(
            x,
            modulation=modulation,
            params=params.attention,
        )
        + x
    )
    return mlp(x, modulation=modulation, params=params.mlp) + x


class DiTParams(typing.NamedTuple):
    proj_in: jax.Array
    fourier_features: FourierFeaturesParams
    layers: list[TransformerLayerParams]
    norm: jax.Array
    proj_out: jax.Array


def init_dit(config: DiTConfig, key: jax.typing.ArrayLike) -> DiTParams:
    keys = jax.random.split(key, config.num_layers + 3)
    return DiTParams(
        proj_in=init_linear(
            config.in_dim, out_dim=config.hidden_dim // config.patch_size, key=keys[0]
        ),
        fourier_features=init_fourier_features(config.time_dim, key=keys[1]),
        layers=[
            init_transformer_layer(
                config.hidden_dim,
                modulation_dim=config.time_dim,
                num_heads=config.num_heads,
                seq_len=config.seq_len,
                rope_base=config.rope_base,
                key=k,
            )
            for k in keys[2:-1]
        ],
        norm=init_layernorm(config.hidden_dim),
        proj_out=init_linear(
            config.hidden_dim // config.patch_size,
            out_dim=config.in_dim,
            key=keys[-1],
            zero=True,
        ),
    )


def dit(x: jax.Array, time: jax.Array, params: DiTParams, config: DiTConfig) -> jax.Array:
    seq_len = x.shape[0]
    x = linear(x, params=params.proj_in)
    x = jnp.reshape(x, shape=(seq_len // config.patch_size, config.hidden_dim))

    modulation = fourier_features(time, params.fourier_features)

    def scan_fn(carry: jax.Array, layer: typing.Any) -> tuple[jax.Array, None]:
        return transformer_layer(carry, modulation=modulation, params=layer), None

    layers_stacked = jax.tree_util.tree_map(
        lambda *args: jnp.stack(args), *params.layers
    )  # TODO: does this cause a slow down?
    x = jax.lax.scan(scan_fn, x, layers_stacked)[0]

    x = layernorm(x, params=params.norm)
    x = jnp.reshape(x, (seq_len, -1))
    return linear(x, params=params.proj_out)


@partial(jax.jit, static_argnums=(1, 2, 3))
def generate(
    dit_params: DiTParams,
    bs: int,
    steps: int,
    config: DiTConfig,
    key: jax.typing.ArrayLike,
) -> jax.Array:
    noise = jax.random.normal(
        key,
        shape=(bs, config.seq_len, config.in_dim),
        dtype=jax.tree.leaves(dit_params)[0].dtype,
    )

    def scan_fn(noise: jax.Array, i: jax.Array) -> tuple[jax.Array, None]:
        t = jnp.full((bs, 1, 1), fill_value=i / steps, dtype=noise.dtype)
        v = jax.vmap(partial(dit, params=dit_params, config=config))(noise, time=t)
        noise = noise - (1.0 / steps) * v
        return noise, None

    return jax.lax.scan(scan_fn, noise, jnp.arange(steps, 0, -1))[0]


if __name__ == "__main__":
    key = jax.random.key(42)
    shape = (8, 100, 16)
    arr = jax.random.normal(key, shape, dtype=jnp.bfloat16)
    time = jax.random.normal(key, (8, 1, 1), dtype=jnp.bfloat16)

    config = DiTConfig()
    dit_params = init_dit(config, key)
    dit_params = jax.tree.map(lambda x: jnp.astype(x, jnp.bfloat16), dit_params)
    out = jax.vmap(partial(dit, params=dit_params, config=config))(arr, time=time)
    generated = generate(dit_params, bs=4, steps=4, config=config, key=key)
    print(out.shape)
