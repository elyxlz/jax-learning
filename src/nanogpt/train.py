from dataclasses import dataclass

from .model import GPTConfig, GPTModel


@dataclass
class TrainConfig:
    model_config: GPTConfig
    learning_rate: float
    batch_size: int


@dataclass
class TrainState:
    model: GPTModel
    opt_s
