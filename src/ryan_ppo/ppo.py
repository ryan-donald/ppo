from __future__ import annotations

import math

import torch
import torch.optim as optim

from ryan_ppo.config import TrainConfig
from ryan_ppo.network import Actor, Critic
from ryan_ppo.normalization import ObsNormalization
from ryan_ppo.storage import RolloutBatch
from ryan_ppo.utils import CapturedStep

KL_LR_DECREASE_FACTOR = 2.0
KL_LR_INCREASE_FACTOR = 2.0
LR_ADJUST_RATIO = 1.5

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)

# performance improvements by adjusting compile mode for minibatch loss.
COMPILE_MODE = "max-autotune-no-cudagraphs"


def strip_compile_prefix(state_dict: dict) -> dict:
    # allows loading of checkpoints saved from a compiled module.
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def gaussian_log_prob(
    x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    # diagonal gaussian log-density, summed over action dims.
    return (-0.5 * ((x - mu) / std).square() - log_std - LOG_SQRT_2PI).sum(dim=-1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    # closed-form diagonal gaussian entropy, summed over action dims.
    return (log_std + 0.5 + LOG_SQRT_2PI).sum()


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: TrainConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        device = torch.device(device)
        if cfg.use_normalization:
            self.obs_normalizer = ObsNormalization(state_dim)
        else:
            self.obs_normalizer = None

        self.device = device

        self.actor = Actor(
            state_dim,
            action_dim,
            cfg.hidden_dims,
            self.obs_normalizer,
            std=cfg.std_init,
        ).to(device)
        self.critic = Critic(state_dim, cfg.hidden_dims, self.obs_normalizer).to(device)

        self.log_std_min = math.log(cfg.std_min)
        self.log_std_max = math.log(cfg.std_max)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        # learning rate stored as tensor for speed.
        self.lr_t = torch.tensor(float(cfg.learning_rate), device=device)

        # on cuda, each minibatch step is replayed as a CUDA graph.
        self.use_cuda_graph_update = device.type == "cuda"
        self.graph_key = None

        self.optimizer = optim.Adam(
            self.actor_params + self.critic_params,
            lr=self.lr_t,
            fused=(device.type == "cuda"),
            foreach=False if device.type != "cuda" else None,
            capturable=self.use_cuda_graph_update,
        )

        # hyperparameters
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_epsilon = cfg.clip_epsilon
        self.max_grad_norm = cfg.max_grad_norm
        self.entropy_coef = cfg.entropy_coef
        self.value_coef = cfg.value_coef
        self.desired_kl = cfg.desired_kl
        self.schedule_type = cfg.schedule_type
        self.max_lr = cfg.max_lr
        self.min_lr = cfg.min_lr

        self.update_count = 0

    @property
    def current_lr(self) -> float:
        # GPU to CPU sync for storing/logging learning rate.
        return self.lr_t.item()

    @current_lr.setter
    def current_lr(self, value: float) -> None:
        self.lr_t.fill_(float(value))

    @torch.compile
    def adapt_lr_device(self, kl: torch.Tensor) -> None:
        # adaptive kl, but done on GPU fully for speed.
        lr = self.lr_t
        adjusted = torch.where(
            kl > self.desired_kl * KL_LR_DECREASE_FACTOR,
            lr / LR_ADJUST_RATIO,
            torch.where(
                (kl > 0.0) & (kl < self.desired_kl / KL_LR_INCREASE_FACTOR),
                lr * LR_ADJUST_RATIO,
                lr,
            ),
        )
        self.lr_t.copy_(adjusted.clamp_(self.min_lr, self.max_lr))

    @torch.compile
    def update_normalization(self, obs: torch.Tensor) -> None:
        self.actor.update_normalization(obs)

    def save_checkpoint(self, path: str, iteration: int) -> None:
        # save a complete checkpoint for resuming training. includes weights,
        # optimizer, lr, and number of updates. allows for seamless saving of
        # checkpoints to be used for resuming later, or for experimenting with
        # fine-tuning.

        torch.save(
            {
                "iteration": iteration,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "current_lr": self.current_lr,
                "update_count": self.update_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        # fully loads the checkpoint saved by the save_checkpoint() function.

        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(strip_compile_prefix(checkpoint["actor"]))
        self.actor.log_std.data.clamp_(min=self.log_std_min, max=self.log_std_max)
        self.critic.load_state_dict(strip_compile_prefix(checkpoint["critic"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_t
            group["capturable"] = self.use_cuda_graph_update
        self.current_lr = checkpoint["current_lr"]
        self.update_count = checkpoint.get("update_count", 0)
        return checkpoint["iteration"]

    def select_action(
        self, state_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # selects action based upon an observation and the current policy,
        # returns action, log_prob, mu, std.
        if not torch.is_tensor(state_obs):
            state_obs = torch.tensor(state_obs, dtype=torch.float, device=self.device)
        else:
            state_obs = state_obs.to(self.device)

        # necessary for new random numbers every replay of the act graph.
        noise = torch.randn(
            state_obs.shape[0], self.actor.log_std.shape[0], device=self.device
        )
        with torch.no_grad():
            return self.act(state_obs, noise)

    @torch.compile
    def act(
        self, state_obs: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # performs forward pass, gaussian sample, and log_prob in one compiled function
        mu, std, log_std = self.actor(state_obs)
        action = torch.addcmul(mu, std, noise)
        log_prob = gaussian_log_prob(action, mu, std, log_std)
        return action, log_prob, mu, std

    @torch.compile
    def act_deterministic(self, state_obs: torch.Tensor) -> torch.Tensor:
        # the policy mean, for evaluation/play.
        mu, _, _ = self.actor(state_obs)
        return mu

    @torch.compile
    def evaluate_values(self, states: torch.Tensor) -> torch.Tensor:
        # critic values with the trailing singleton dim dropped.
        return self.critic(states).squeeze(-1)

    def entropy(self) -> float:
        with torch.no_grad():
            return gaussian_entropy(self.actor.log_std).item()

    @torch.compile
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # computes generalized advantage estimates (GAE)

        values_extended = torch.cat([values, next_value.unsqueeze(0)], dim=0)

        deltas = (
            rewards
            + self.gamma * values_extended[1:] * (1 - dones)
            - values_extended[:-1]
        )

        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)

        for step in reversed(range(num_steps)):
            gae = deltas[step] + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae

        returns = advantages + values

        return advantages, returns

    @torch.compile(mode=COMPILE_MODE)
    def minibatch_loss(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_log_probs_old: torch.Tensor,
        batch_returns: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_values_old: torch.Tensor,
        batch_mus_old: torch.Tensor,
        std_old: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # single minibatch loss and kl in a single compiled function.

        batch_states = batch_states[indices]
        batch_actions = batch_actions[indices]
        batch_log_probs_old = batch_log_probs_old[indices]
        batch_returns = batch_returns[indices]
        batch_advantages = batch_advantages[indices]
        batch_values_old = batch_values_old[indices]
        batch_mus_old = batch_mus_old[indices]

        # calculate log_probs for current policy.
        mu, std, log_std = self.actor(batch_states)

        log_probs = gaussian_log_prob(batch_actions, mu, std, log_std)
        entropy = gaussian_entropy(log_std)

        # full KL divergence.
        mu_d = mu.detach()
        std_d = std.detach()
        log_ratio = std_old.log() - log_std.detach()
        kl = (0.5 * torch.expm1(2.0 * log_ratio) - log_ratio).sum() + (
            0.5 * ((batch_mus_old - mu_d) / std_d).square()
        ).sum(dim=-1).mean()

        # compute surrogate loss
        ratios = torch.exp(log_probs - batch_log_probs_old)
        surr1 = ratios * batch_advantages
        surr2 = (
            torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * batch_advantages
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        # compute clipped value loss
        values = self.critic(batch_states).view(-1)
        value_pred_clipped = batch_values_old + torch.clamp(
            values - batch_values_old, -self.clip_epsilon, self.clip_epsilon
        )
        value_losses = (values - batch_returns).pow(2)
        value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
        critic_loss = torch.max(value_losses, value_losses_clipped).mean()

        # total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        return loss, kl

    def update(
        self,
        batch: RolloutBatch,
        epochs: int = 4,
        num_mini_batches: int = 4,
    ) -> float:
        # updates Actor and Critic networks using the PPO algorithm.

        # normalize advantages once over the full batch
        advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
        )

        dataset_size = len(batch)
        batch_size = dataset_size // num_mini_batches

        data = (
            batch.states,
            batch.actions,
            batch.log_probs_old,
            batch.returns,
            advantages,
            batch.values_old,
            batch.mus_old,
            batch.std_old,
        )

        # KL accumulates on device and the adaptive schedule runs on device, so the
        # whole update needs no GPU to CPU sync until the mean is read out below.
        if self.use_cuda_graph_update:
            kl_sum = self.load_graph_batch(data, batch_size)
        else:
            kl_sum = torch.zeros((), device=self.device)
        num_updates = 0

        # training loop
        for _ in range(epochs):
            # randomizes batch data
            indices = torch.randperm(dataset_size, device=self.device)

            # mini-batch updates, dropping the remainder rows.
            for start in range(0, batch_size * num_mini_batches, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                if self.use_cuda_graph_update:
                    self.graph_indices.copy_(batch_indices)
                    self.graph_step()
                else:
                    self.minibatch_step(data, batch_indices, kl_sum)
                num_updates += 1

        # average KL divergence over all minibatch updates.
        mean_kl = (kl_sum / num_updates).item()
        if not math.isfinite(mean_kl):
            raise RuntimeError(f"KL is non-finite at update {self.update_count}")

        self.update_count += 1

        return mean_kl

    def minibatch_step(
        self,
        data: tuple[torch.Tensor, ...],
        indices: torch.Tensor,
        kl_sum: torch.Tensor,
    ) -> None:
        # one gradient step, with no GPU to CPU sync so it can be captured.
        loss, kl = self.minibatch_loss(*data, indices)

        # gradient descent step, with a clipped gradient norm. called here
        # to queue calculation on gpu while the kl accumulation below is
        # enqueued.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        kl = kl.detach()
        kl_sum += kl
        if self.schedule_type == "adaptive":
            self.adapt_lr_device(kl)

        # clip actor and critic norms seperately, as they can interfere.
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
        self.optimizer.step()

        # clamp log_std post optimizer to keep gradients useful
        self.actor.log_std.data.clamp_(min=self.log_std_min, max=self.log_std_max)

    def load_graph_batch(
        self, data: tuple[torch.Tensor, ...], batch_size: int
    ) -> torch.Tensor:
        # a graph replays on fixed memory, so the batch is copied into it. a new batch
        # shape captures a new graph.
        key = (tuple(t.shape for t in data), batch_size)
        if key != self.graph_key:
            self.graph_key = key
            self.graph_data = tuple(torch.empty_like(t) for t in data)
            self.graph_indices = torch.empty(
                batch_size, dtype=torch.long, device=self.device
            )
            self.graph_kl_sum = torch.zeros((), device=self.device)
            self.graph_step = CapturedStep(
                lambda: self.minibatch_step(
                    self.graph_data, self.graph_indices, self.graph_kl_sum
                )
            )

        for buffer, t in zip(self.graph_data, data):
            buffer.copy_(t)
        self.graph_kl_sum.zero_()
        return self.graph_kl_sum
