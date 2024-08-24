from dataclasses import dataclass
import math
from pdb import set_trace as st
import jax
import equinox as eqx
import jax.numpy as jnp
from tqdm import trange


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    eps: float = 1e-5
    hidden_size: int = 128


def nearest_128(n: int) -> int:
    return int(128 * round(n / 128))


class RMSNorm(eqx.Module):
    eps: float
    weight: jax.Array

    def __init__(self, dim: int, eps: float, dtype: jnp.dtype = jnp.bfloat16) -> None:
        self.eps = eps
        self.weight = jnp.ones(shape=dim, dtype=dtype)

    def _norm(self, x: jax.Array) -> jax.Array:
        return x * jax.lax.rsqrt(jnp.mean(x**2, keepdims=True) + self.eps)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._norm(x.astype(jnp.float32)).astype(x.dtype)


class FeedForward(eqx.Module):
    norm: RMSNorm
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(
        self,
        config: GPTConfig,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm = RMSNorm(config.hidden_size, eps=config.eps)
        intermediate_size = nearest_128(config.hidden_size * 4)
        self.w1 = eqx.nn.Linear(
            config.hidden_size,
            intermediate_size,
            use_bias=False,
            key=k1,
            dtype=dtype,
        )
        self.w2 = eqx.nn.Linear(
            intermediate_size,
            config.hidden_size,
            use_bias=False,
            key=k2,
            dtype=dtype,
        )
        self.w3 = eqx.nn.Linear(
            config.hidden_size,
            intermediate_size,
            use_bias=False,
            key=k3,
            dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        return jax.vmap(self.w2)(
            jax.nn.silu(jax.vmap(self.w1)(x)) * jax.vmap(self.w3)(x)
        )


class Attention(eqx.Module):
    n_head: int
    hidden_size: int
    norm: RMSNorm
    mask: jax.Array
    wqkv: eqx.nn.Linear
    wo: eqx.nn.Linear

    def __init__(
        self,
        config: GPTConfig,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.hidden_size = config.hidden_size
        self.n_head = config.n_head

        k1, k2 = jax.random.split(key, 2)
        self.norm = RMSNorm(config.hidden_size, eps=config.eps, dtype=dtype)
        self.wqkv = eqx.nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            dtype=dtype,
            use_bias=False,
            key=k1,
        )
        self.wo = eqx.nn.Linear(
            config.hidden_size, config.hidden_size, dtype=dtype, use_bias=False, key=k2
        )

        mask = jnp.tril(
            jnp.full(
                (config.block_size, config.block_size), dtype=jnp.bfloat16, fill_value=1
            ),
            k=0,
        )

        self.mask = jnp.log(jnp.triu(mask, k=-config.block_size))

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        seq_len = x.shape[0]

        x = jax.vmap(self.norm)(x)

        xqkv = jax.vmap(self.wqkv)(x)
        xqkv = jnp.reshape(
            xqkv,
            (
                seq_len,
                self.n_head,
                self.hidden_size * 3 // self.n_head,
            ),
        )
        xq, xk, xv = jnp.split(xqkv, 3, axis=2)

        # (s h d) @ (s h d)
        scores = (xq.transpose(1, 0, 2) @ xk.transpose(1, 2, 0)) * (
            1.0 / math.sqrt(xk.shape[2])
        )
        scores = scores + self.mask[jnp.newaxis, :seq_len, :seq_len]
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x)
        # (h s s) @ (s h d)
        out = (scores @ xv.transpose(1, 0, 2)).transpose(1, 0, 2)
        out = jnp.reshape(out, (seq_len, -1))

        return jax.vmap(self.wo)(out)


class GPTLayer(eqx.Module):
    attn: Attention
    ff: FeedForward

    def __init__(
        self,
        config: GPTConfig,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        k1, k2 = jax.random.split(key, 2)
        self.attn = Attention(config, key=k1, dtype=dtype)
        self.ff = FeedForward(config, key=k2, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.attn(x) + x
        return self.ff(x) + x


class GPTModel(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: list[GPTLayer]
    final_norm: RMSNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        config: GPTConfig,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        keys = jax.random.split(key, config.n_layer + 2)
        embedding_key, head_key, layer_keys = keys[0], keys[1], keys[2:]
        self.embedding = eqx.nn.Embedding(
            config.vocab_size, config.hidden_size, key=embedding_key, dtype=dtype
        )

        # self.layers = [GPTLayer(config, key=k, dtype=dtype) for k in layer_keys]
        make_layer = lambda k: GPTLayer(config, key=k, dtype=dtype)  # noqa
        self.layers = eqx.filter_vmap(make_layer)(layer_keys)

        self.final_norm = RMSNorm(config.hidden_size, eps=config.eps, dtype=dtype)
        self.head = eqx.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            key=head_key,
            dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.vmap(self.embedding)(x)
        h = self.final_norm(h)

        dynamic_layers, static_layers = eqx.partition(self.layers, eqx.is_array)

        def apply_layer(h: jax.Array, dynamic_layer: GPTLayer) -> jax.Array:
            layer = eqx.combine(dynamic_layer, static_layers)
            return layer(h), None  # type: ignore

        h, _ = jax.lax.scan(apply_layer, h, dynamic_layers)  # type: ignore

        return jax.vmap(self.head)(h).astype(jnp.float32)

    @eqx.filter_jit()
    def generate(
        self, initial_input: jax.Array, max_length: int, temperature: float = 1.0
    ) -> jax.Array:
        generated = initial_input

        for _ in trange(max_length):
            output = self(generated)
            logits = output[:, -1]
            logits /= temperature
            next_token = jax.random.categorical(jax.random.PRNGKey(0), logits, axis=-1)
            generated = jnp.concatenate(
                [generated, jnp.expand_dims(next_token, 0)], axis=0
            )
        return generated


if __name__ == "__main__":
    config = GPTConfig()
    key = jax.random.PRNGKey(0)
    gpt = GPTModel(config, key=key, dtype=jnp.bfloat16)
    tensor = jax.random.randint(
        jax.random.PRNGKey(1), (64,), minval=0, maxval=config.vocab_size
    )
    out = gpt(tensor)
    print(f"forward: {out.shape}")
    generated = gpt.generate(tensor, max_length=64)
    print(f"generated: {generated.shape}")
    tensor2 = jax.random.randint(
        jax.random.PRNGKey(2), (64,), minval=0, maxval=config.vocab_size
    )
    generated_again = gpt.generate(tensor2, max_length=64)
    print(f"generated_again: {generated_again.shape}")
