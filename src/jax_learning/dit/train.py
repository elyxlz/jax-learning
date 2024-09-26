import functools
import typing
import rich
import time

import jax
import jax.numpy as jnp
import optax
import tqdm

from .model import DiTConfig, DiTParams, dit, init_dit


class TrainConfig(typing.NamedTuple):
    name: str = "a-dit-run"
    seed: int = 42
    dit_config: DiTConfig = DiTConfig()
    lr: float = 3e-4
    betas: tuple[float, float] = (0.95, 0.99)
    eps: float = 1e-11
    weight_decay: float = 0.1
    grad_norm: float = 0.3
    max_steps: int = 1_000

    save_every: int = 5_000


class TrainState(typing.NamedTuple):
    step: int
    dit_params: DiTParams
    opt_state: optax.OptState


console = rich.console.Console()


def pprint(
    item: str | dict, *args, color: str | None = None, json: bool = False, **kwargs
) -> None:
    if json:
        console.print_json(data=item, *args, **kwargs)
    else:
        item = f"[{color}]{item}[/{color}]" if color else item
        console.print(item, *args, **kwargs)


def init_train_state(config: TrainConfig) -> TrainState:
    dit_params = init_dit(config.dit_config, key=jax.random.key(config.seed))
    opt_state = optax.adamw(
        config.lr,
        b1=config.betas[0],
        b2=config.betas[1],
        eps=config.eps,
        weight_decay=config.weight_decay,
        mask=jax.tree.map(lambda x: jnp.ndim(x) > 1, dit_params),
    ).init(dit_params)

    return TrainState(step=0, dit_params=dit_params, opt_state=opt_state)


def compute_loss(
    batch: dict, dit_params: DiTParams, step: int, dit_config: DiTConfig
) -> jax.Array:
    batch_size = batch["x"].size(0)

    t = jax.random.uniform(jax.random.key(step), shape=(batch_size,), dtype=jnp.bfloat16)
    texp = t.view([batch_size, *([1] * len(batch["x"].shape[1:]))])
    z1 = jax.random.normal(jax.random.key(step), shape=batch["x"].shape, dtype=jnp.bfloat16)
    zt = (1 - texp) * batch["x"] + texp * z1
    out = dit(zt, time=t, params=dit_params, config=dit_config)
    return (out.astype(jnp.float32) - (z1 - batch["x"]).astype(jnp.float32)) ** 2


@jax.jit
def train_step(batch: dict, train_state: TrainState, config: TrainConfig) -> TrainState:
    loss, grad = jax.value_and_grad(
        functools.partial(
            compute_loss,
            dit_params=train_state.dit_params,
            step=train_state.step,
            dit_config=config.dit_config,
        )
    )(batch)

    optim = optax.adamw(
        config.lr,
        b1=config.betas[0],
        b2=config.betas[1],
        eps=config.eps,
        weight_decay=config.weight_decay,
        mask=jax.tree.map(lambda x: jnp.ndim(x) > 1, train_state.dit_params),
    )

    updates, new_opt_state = optim.update(
        grad, state=train_state.opt_state, params=train_state.dit_params
    )
    new_dit_params = optax.apply_updates(train_state.dit_params, updates=updates)

    return TrainState(step=train_state.step, dit_params=new_dit_params, opt_state=new_opt_state)


def train(config: TrainConfig) -> None:
    state = init_train_state(config)

    train_bar = tqdm.trange(
        config.max_steps - state.step, initial=state.step, total=config.max_steps, colour="blue"
    )

    for _ in train_bar:
        start_time = time.time()
        try:
            batch = next(train_loader)
        except StopIteration:
            continue

        state.step += 1
        state, stats, ema_info = train_step(state, batch=batch, stats=stats, config=config)

        end_time = time.time()
        throughput = calculate_throughput(
            end_time - start_time, batch=batch, hz=config.codec_hz
        )

        # log
        train_bar.set_description(f"loss: {stats[0]/stats[-1]:.2f} thr: {throughput:.0f}")
        utils.rank_0_only(wandb.log)(
            ema_info, step=state.step
        ) if ema_info is not None else None
        if state.step % 50 == 0:
            stats = log_metrics(state, stats)

        if config.save_every and state.step % config.save_every == 0:
            utils.rank_0_only(save_state)(state, config=config)

        if config.push_every and state.step % config.push_every == 0:
            utils.rank_0_only(state.ema.push_to_hub)(
                f"Audiogen/{config.name}",
                commit_message=f"step {state.step}, run_id {config.run_id}",
                private=True,
            )

        if config.val_every and state.step % config.val_every == 0:
            validation(state.ema, step=state.step, config=config)

        if config.test_every and state.step % config.test_every == 0:
            for c in utils.unwrap_model(state.model).config.conditions:
                test(c.name, ema_model=state.ema, step=state.step, config=config)

        if state.step >= config.max_steps:
            utils.pprint("\nmax steps reached, exiting...", color="bold red")
            break

        utils.distributed_only(dist.barrier)()  # type: ignore


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
