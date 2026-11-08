import configparser
import math
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

# configparser reader per field type.
TYPE_READERS = {
    "float": "getfloat",
    "int": "getint",
    "bool": "getboolean",
    "str": "get",
}


@dataclass
class TrainConfig:
    """parses config file for tasks."""

    learning_rate: float
    gamma: float
    gae_lambda: float
    value_coef: float
    clip_epsilon: float
    max_grad_norm: float
    desired_kl: float
    entropy_coef: float
    schedule_type: str
    num_learning_epochs: int
    num_steps_per_env: int
    num_mini_batches: int
    max_iterations: int
    use_normalization: bool
    hidden_dims: list[int]
    max_lr: float = 1e-3
    min_lr: float = 1e-5
    value_clip: float = math.inf
    std_min: float = 0.005
    std_max: float = math.exp(2.0)
    std_init: float = 0.5
    stagger_initial_episodes: bool = True

    @classmethod
    def from_ini(cls, path):
        # allows reading a default config then a override, or a full config.
        path = Path(path)
        config = configparser.ConfigParser()
        config.read([path.parent / "defaults.ini", path])
        train = config["train"]

        kwargs = {}
        for field in fields(cls):
            if field.name == "hidden_dims":
                continue
            type_name = getattr(field.type, "__name__", field.type)
            read = getattr(train, TYPE_READERS[type_name])
            assert field.default is not MISSING or field.name in train, (
                f"{path}: required [train] key {field.name!r} is missing from "
                f"both the task file and defaults.ini"
            )
            kwargs[field.name] = read(field.name, fallback=field.default)

        hidden_dims = [int(x) for x in config["policy"]["hidden_dims"].split(",")]
        return cls(hidden_dims=hidden_dims, **kwargs)

    def apply_sweep(self, sweep):
        # sets each swept parameter on the config field of the same name.
        names = {field.name for field in fields(self)}
        for key, value in sweep.items():
            assert key in names, f"sweep parameter {key!r} is not a TrainConfig field"
            setattr(self, key, value)
