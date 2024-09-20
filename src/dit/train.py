import jax
import jax.numpy as jnp

import typing
import optax
import tqdm

from .model import DiTParams, init_dit, dit, DiTConfig


class TrainConfig(typing.NamedTuple):
    seed: int = 42
    dit_config: DiTConfig = DiTConfig()
    lr: float = 3e-4
    betas: tuple[float, float] = (0.95, 0.99)
    eps: float = 1e-11
    weight_decay: float = 0.1
    grad_norm: float = 0.3
    max_steps: int = 1_000


class TrainState(typing.NamedTuple):
    dit_params: DiTParams
    opt_state: optax.OptState


def init_train_state(config: TrainConfig) -> TrainState:
    dit_params = init_dit(config.dit_config, key=jax.random.key(config.seed))
    optim = optax.adamw(
        config.lr,
        b1=config.betas[0],
        b2=config.betas[1],
        eps=config.eps,
        weight_decay=config.weight_decay,
        mask=jax.tree.map(lambda x: jnp.ndim(x) > 1, dit_params),
    )
    opt_state = optim.init(dit_params)

    return TrainState(dit_params=dit_params, opt_state=opt_state)


def train_step(dit_params: DiTParams, train_state: TrainState, batch: dict) -> TrainState:
    raise NotImplementedError()


def train(config: TrainConfig) -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
