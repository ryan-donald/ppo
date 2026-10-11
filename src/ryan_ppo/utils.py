from __future__ import annotations

import random
import sys
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import numpy as np
import rich.traceback
import torch

# libraries to suppress in rich traceback.
SUPPRESSED_LIBRARIES = ("gymnasium",)


def install_rich_traceback(show_locals: bool = False):
    """
    initialize the rich traceback, including suppressing specified libaries.
    """
    # every isaaclab package, when running against isaaclab at all
    suppress = []
    isaaclab = sys.modules.get("isaaclab")
    if isaaclab is not None:
        suppress.append(str(Path(isaaclab.__file__).parents[2]))

    suppress += [import_module(name) for name in SUPPRESSED_LIBRARIES]

    rich.traceback.install(suppress=suppress, show_locals=show_locals)
    return sys.excepthook


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    return device


class PhaseTimer:
    """Time a phase on its CUDA stream, or with a CPU clock on CPU.

    Read seconds() after the update's existing synchronization, so recording
    phase boundaries does not introduce synchronization between phases.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed = 0.0

    def start(self) -> None:
        if self.cuda:
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> None:
        if self.cuda:
            self.end_event.record(torch.cuda.current_stream(self.device))
        else:
            self.elapsed = time.perf_counter() - self.start_time

    def seconds(self) -> float:
        if self.cuda:
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        return self.elapsed


class CapturedStep:
    """
    calls fn eagerly for its first calls, then captures it as a CUDA graph and
    replays it. fn must only use fixed memory. pool shares a graph memory pool.
    """

    def __init__(self, fn, warmup_steps: int = 3, pool=None) -> None:
        self.fn = fn
        self.warmup_left = warmup_steps
        self.pool = pool
        self.graph = None

    def __call__(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        elif self.warmup_left > 0:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.fn()
            torch.cuda.current_stream().wait_stream(stream)
            self.warmup_left -= 1
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self.fn()
            # capture only records the kernels, so replay to actually take this step.
            graph.replay()
            self.graph = graph


def set_seed(seed: int) -> None:
    # seed python, numpy and torch, and pin cudnn to deterministic kernels.
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_dims(env) -> tuple[int, int]:
    # (state_dim, action_dim), for both Dict and Box observation spaces.
    obs_space = env.observation_space
    spaces = getattr(obs_space, "spaces", None)
    if spaces is not None:
        obs_space = spaces.get("policy", next(iter(spaces.values())))
    return obs_space.shape[1], env.action_space.shape[1]


class Profiler:
    """carb.profiler zones, or no-ops when profiling is off.

    carb is only imported when profiling is enabled.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.carb = None
        if enabled:
            import carb.profiler

            self.carb = carb.profiler

    def begin(self, zone_id: int, name: str) -> None:
        if self.carb:
            self.carb.begin(zone_id, name)

    def end(self, zone_id: int) -> None:
        if self.carb:
            self.carb.end(zone_id)

    @contextmanager
    def zone(self, zone_id: int, name: str):
        self.begin(zone_id, name)
        try:
            yield
        finally:
            self.end(zone_id)


def policy_obs(state):
    # returns observation regardless of if it is in a dict or a box.
    if isinstance(state, dict):
        return state.get("policy", next(iter(state.values())))
    return state


def warmup_normalization(env, agent, state, num_steps: int):
    # warm-up the observation normalizer with a full rollout of observations.
    obs = []
    for _ in range(num_steps):
        obs.append(policy_obs(state).clone())
        state, *_ = env.step(agent.select_action(policy_obs(state))[0])
    agent.update_normalization(torch.cat(obs))
    return state


def get_cfg_path(task):
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    ini_file_path = project_root / "cfg" / f"{task}.ini"
    if not ini_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {ini_file_path}")

    return ini_file_path
