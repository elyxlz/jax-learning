from dataclasses import dataclass
import math

import jax
import equinox as eqx
import jax.numpy as jnp


@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    eps: float = 1e-5
    hidden_size: int = 128


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
    config: GPTConfig
    norm: RMSNorm
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    # TODO: 8/3 aspect ratio
    def __init__(
        self,
        config: GPTConfig,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config

        k1, k2, k3 = jax.random.split(key, 3)
        self.norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.w1 = eqx.nn.Linear(
            config.hidden_size,
            config.hidden_size * 4,
            use_bias=False,
            key=k1,
            dtype=dtype,
        )
        self.w2 = eqx.nn.Linear(
            config.hidden_size * 4,
            config.hidden_size,
            use_bias=False,
            key=k2,
            dtype=dtype,
        )
        self.w3 = eqx.nn.Linear(
            config.hidden_size,
            config.hidden_size * 4,
            use_bias=False,
            key=k3,
            dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        return jax.vmap(self.w2)(
            jax.nn.silu(jax.vmap(self.w1)(x)) * jax.vmap(self.w3)(x)
        )


class CausalSelfAttention(eqx.Module):
    config: GPTConfig
    norm: RMSNorm
    wqkv: eqx.nn.Linear
    wo: eqx.nn.Linear

    def __init__(
        self,
        config: GPTConfig,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config

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

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        seq_len = x.shape[0]

        x = jax.vmap(self.norm)(x)

        xqkv = jax.vmap(self.wqkv)(x)
        xqkv = jnp.reshape(
            xqkv,
            (
                seq_len,
                self.config.n_head,
                self.config.hidden_size * 3 // self.config.n_head,
            ),
        )
        xq, xk, xv = jnp.split(xqkv, 3, axis=2)

        scores = jnp.matmul(xq, xk.transpose(0, 2, 1)) * (1.0 / math.sqrt(xk.shape[2]))
        if mask is not None:
            scores = scores + mask[jnp.newaxis, ...]
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x)
        out = jnp.matmul(scores, xv)
        out = jnp.reshape(out, (seq_len, -1))

        return jax.vmap(self.wo)(out)


class GPTLayer(eqx.Module):
    config: GPTConfig
    attn: CausalSelfAttention
    ff: FeedForward

    def __init__(
        self,
        config: GPTConfig,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config

        k1, k2 = jax.random.split(key, 2)
        self.attn = CausalSelfAttention(config, key=key, dtype=dtype)
        self.ff = FeedForward(config, key=key, dtype=dtype)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        x = self.attn(x) + x
        return self.ff(x) + x


class GPTModel(eqx.Module):
    embedding: eqx.nn.Embedding
    mask: jax.Array
    layers: list[GPTLayer]
    final_norm: RMSNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        config: GPTConfig,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        keys = jax.random.split(key, config.n_layer + 2)
        embedding_key, head_key, layer_keys = keys[0], keys[1], keys[2:]
        self.embedding = eqx.nn.Embedding(
            config.vocab_size, config.hidden_size, key=embedding_key, dtype=dtype
        )

        make_layers = lambda k: GPTLayer(config, key=k, dtype=dtype)
        self.layers = eqx.filter_vmap(make_layers)(layer_keys)
        del make_layers

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
        for layer in self.layers:
            h = layer(h, mask=self.mask)

        return jax.vmap(self.head)(h)

    def generate(
        self, initial_input: jax.Array, max_length: int, temperature: float = 1.0
    ) -> jax.Array:
        generated = jnp.expand_dims(initial_input, axis=1)

        for _ in range(max_length):
            output = self(generated)
            logits = output[:, -1, :]
            logits /= temperature
            next_token = jax.random.categorical(jax.random.PRNGKey(0), logits, axis=-1)
            generated = jnp.concatenate(
                [generated, jnp.expand_dims(next_token, axis=1)], axis=1
            )

        return generated


if __name__ == "__main__":
    config = GPTConfig()
    key = jax.random.PRNGKey(0)
    gpt = GPTModel(config, key=key, dtype=jnp.bfloat16)
    tensor = jax.random.normal(jax.random.PRNGKey(1), (64, 128))
    out = gpt(tensor)
