import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ryan_ppo.tracking
import ryan_ppo.training
from ryan_ppo.training import train_agent
from ryan_ppo.utils import Profiler, set_seed

NUM_STEPS_PER_ENV = 16

TASK_INI = f"""
[train]
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
value_coef = 1.0
clip_epsilon = 0.2
max_grad_norm = 1.0
desired_kl = 0.01
entropy_coef = 0.0
schedule_type = adaptive
num_learning_epochs = 5
num_steps_per_env = {NUM_STEPS_PER_ENV}
num_mini_batches = 4
max_iterations = 40
use_normalization = True

[policy]
hidden_dims = 32,32
"""


class FakeReachEnv:
    """
    minimal manager-based env: the action should match a 2d target that is in the
    observation, held for the episode. a random policy scores ~-30 per episode.
    """

    def __init__(self, num_envs: int = 64) -> None:
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = num_envs
        self.observation_space = SimpleNamespace(shape=(num_envs, 4))
        self.action_space = SimpleNamespace(shape=(num_envs, 2))
        self.max_episode_length = 32
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.common_step_counter = 0
        self.step_dt = 0.02
        self.reward_manager = SimpleNamespace(
            active_terms=["reach"], _step_reward=torch.zeros(num_envs, 1)
        )
        self.target = torch.zeros(num_envs, 2)

    def obs(self) -> dict:
        # target plus two pure-noise dims.
        return {"policy": torch.cat([self.target, torch.randn_like(self.target)], -1)}

    def reset(self):
        self.target = torch.rand(self.num_envs, 2) * 2 - 1
        return self.obs(), {}

    def step(self, action):
        reward = -(action - self.target).square().sum(-1)
        # per-term rewards are reward rates, as in the isaaclab reward manager.
        self.reward_manager._step_reward.copy_(reward[:, None] / self.step_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        truncated = self.episode_length_buf >= self.max_episode_length
        terminated = torch.rand(self.num_envs) < 0.01
        done = truncated | terminated
        self.target = torch.where(
            done[:, None], torch.rand_like(self.target) * 2 - 1, self.target
        )
        self.episode_length_buf[done] = 0
        return self.obs(), reward, terminated, truncated, {}


@pytest.fixture
def logged(tmp_path, monkeypatch):
    # runs train_agent offline in tmp_path, returning the dicts sent to wandb.log.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    ini = tmp_path / "task.ini"
    ini.write_text(TASK_INI)
    monkeypatch.setattr(ryan_ppo.training, "get_cfg_path", lambda task: ini)

    # wandb.init() rebinds wandb.log, so patch the logger's own module reference.
    records = []
    monkeypatch.setattr(
        ryan_ppo.tracking,
        "wandb",
        SimpleNamespace(log=lambda d, step: records.append({"step": step, **d})),
    )

    # train_agent installs its own SIGINT handler.
    sigint = signal.getsignal(signal.SIGINT)
    yield records
    signal.signal(signal.SIGINT, sigint)


def make_args(**overrides):
    args = dict(
        task="Fake-Reach",
        seed=0,
        max_iterations=None,
        sweep=False,
        save=False,
        resume=None,
        checkpoint_iters="",
    )
    return SimpleNamespace(**{**args, **overrides})


def run(args, env=None):
    set_seed(args.seed)
    env = env or FakeReachEnv()
    train_agent(env, args, "test", Profiler(enabled=False))
    return env


def test_train_agent_learns_fake_reach(logged):
    run(make_args())

    rewards = [d["train/avg_reward"] for d in logged]
    assert len(rewards) == 40
    # random policy ~-30; seeds 0-4 all finish near -5.5.
    final = sum(rewards[-5:]) / 5
    assert final > -12, rewards


def test_train_agent_resumes_from_checkpoint(logged):
    run(make_args(max_iterations=3, save=True, checkpoint_iters="2"))
    (run_dir,) = Path("ppo_logs/Fake-Reach").iterdir()
    assert (run_dir / "checkpoint_2.pth").exists()
    assert (run_dir / "checkpoint_final.pth").exists()
    logged.clear()

    env = run(make_args(max_iterations=4, resume=str(run_dir / "checkpoint_2.pth")))

    # picks up at iteration 2, with the env step counter restored for curricula.
    assert [d["step"] for d in logged] == [2, 3]
    assert env.common_step_counter == 4 * NUM_STEPS_PER_ENV


class FakeDirectEnv:
    # the same task without a reward manager, as in the direct workflow.
    def __init__(self) -> None:
        self.env = FakeReachEnv()
        self.unwrapped = self

    def __getattr__(self, name):
        if name == "reward_manager":
            raise AttributeError(name)
        return getattr(self.env, name)


def test_train_agent_runs_without_a_reward_manager(logged):
    run(make_args(max_iterations=3), env=FakeDirectEnv())

    assert [d["step"] for d in logged] == [0, 1, 2]
    assert not any(key.startswith("rewards/") for key in logged[-1])


class InterruptedEnv(FakeReachEnv):
    # sends ctrl+c partway through the second iteration's rollout, counting the
    # normalizer warm-up rollout and the first iteration before it.
    def step(self, action):
        if self.common_step_counter == 2 * NUM_STEPS_PER_ENV + 5:
            signal.raise_signal(signal.SIGINT)
        return super().step(action)


def test_ctrl_c_saves_after_finishing_the_iteration(logged):
    with pytest.raises(KeyboardInterrupt):
        run(make_args(max_iterations=10, save=True), env=InterruptedEnv())

    # the interrupted iteration still finishes and is logged, then saved.
    assert [d["step"] for d in logged] == [0, 1]
    (run_dir,) = Path("ppo_logs/Fake-Reach").iterdir()
    assert torch.load(run_dir / "checkpoint_latest.pth")["iteration"] == 2
