from __future__ import annotations

import torch

from ryan_ppo.ppo import PPOAgent
from ryan_ppo.storage import RolloutStorage
from ryan_ppo.utils import CapturedStep


class RolloutStepper:
    """
    samples actions and records transitions into storage. on cuda both are replayed
    as CUDA graphs, reading and writing the fixed buffers below.
    """

    def __init__(
        self, agent: PPOAgent, storage: RolloutStorage, step_dt: float
    ) -> None:
        self.agent = agent
        self.storage = storage
        self.step_dt = step_dt
        num_envs = storage.num_envs
        state_dim = storage.states.shape[-1]
        action_dim = storage.actions.shape[-1]
        num_terms = storage.term_rewards.shape[-1]
        device = storage.device

        self.state = torch.zeros(num_envs, state_dim, device=device)
        self.noise = torch.zeros(num_envs, action_dim, device=device)
        self.action = torch.zeros(num_envs, action_dim, device=device)
        self.log_prob = torch.zeros(num_envs, device=device)
        self.mu = torch.zeros(num_envs, action_dim, device=device)
        # state independent, so the same for the whole rollout.
        self.std = torch.zeros(action_dim, device=device)
        self.reward = torch.zeros(num_envs, device=device)
        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.term_reward = torch.zeros(num_envs, num_terms, device=device)

        # storage indexes with a python int, so each step gets its own graph.
        self.act_step = self.compute_action
        self.store_steps = [
            lambda step=step: self.store_step(step) for step in range(storage.num_steps)
        ]
        if torch.device(device).type == "cuda":
            self.act_step = CapturedStep(self.act_step)
            self.store_steps = [CapturedStep(fn) for fn in self.store_steps]

    def act(self, state_obs: torch.Tensor) -> torch.Tensor:
        # the returned action is overwritten by the next call.
        self.state.copy_(state_obs)
        self.noise.normal_()
        self.act_step()
        return self.action

    def record(
        self,
        step: int,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        term_reward: torch.Tensor,
    ) -> None:
        self.reward.copy_(reward)
        self.terminated.copy_(terminated)
        self.truncated.copy_(truncated)
        self.term_reward.copy_(term_reward)
        self.store_steps[step]()

    def compute_action(self) -> None:
        with torch.no_grad():
            action, log_prob, mu, std = self.agent.act(self.state, self.noise)
        self.action.copy_(action)
        self.log_prob.copy_(log_prob)
        self.mu.copy_(mu)
        self.std.copy_(std)

    def store_step(self, step: int) -> None:
        # store steps where envs finished, either terminated or truncated
        done = torch.logical_or(self.terminated, self.truncated)
        self.storage.add(
            step,
            state=self.state,
            action=self.action,
            log_prob=self.log_prob,
            reward=self.reward,
            done=done.float(),
            trunc=self.truncated.float(),
            term_reward=self.term_reward * self.step_dt,
            mu=self.mu,
        )
