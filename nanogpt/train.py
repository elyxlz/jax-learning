from dataclasses import dataclass

from optax import adamw

from .model import GPTModel, GPTConfig


@dataclass
class TrainConfig:
    model_config: GPTConfig
    learning_rate: float
    batch_size: int


@dataclass
class TrainState:
    model: GPTModel
