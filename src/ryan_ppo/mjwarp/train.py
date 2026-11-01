import argparse
import signal
import sys
from datetime import datetime

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from ryan_ppo.checkpointing import CheckpointSaver
from ryan_ppo.config import TrainConfig
from ryan_ppo.ppo import PPOAgent
from ryan_ppo.rollout import RolloutStepper
from ryan_ppo.storage import RolloutStorage
from ryan_ppo.tracking import EpisodeTracker, TrainingLogger
from ryan_ppo.utils import (
    PhaseTimer,
    env_dims,
    get_cfg_path,
    get_device,
    install_rich_traceback,
    policy_obs,
    set_seed,
    warmup_normalization,
)


def train(args_cli):
    # the suppressed-library list lives in utils.install_rich_traceback.
    rich_excepthook = install_rich_traceback()

    torch.set_float32_matmul_precision("high")

    set_seed(args_cli.seed)

    env_cfg = load_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # create environment
    env = ManagerBasedRlEnv(env_cfg, device=args_cli.device)
    device = get_device(env.unwrapped.device)

    # get environment-specific training configuration
    cfg = TrainConfig.from_ini(get_cfg_path(args_cli.task))

    if args_cli.max_iterations is not None:
        cfg.max_iterations = args_cli.max_iterations

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(
        project="PPO MuJoCo Warp",
        name=f"{args_cli.task}_{run_id}",
        settings={"console": "off"},
    )

    # wandb.init() replaces sys.excepthook; put rich's back or a failure prints
    # the plain traceback first and the rich one second.
    sys.excepthook = rich_excepthook

    if args_cli.sweep:
        cfg.apply_sweep(wandb.config)

    state_dim, action_dim = env_dims(env)

    # initialize PPO agent
    agent = PPOAgent(state_dim, action_dim, cfg, device=device)

    # reset environment
    state, info = env.reset()
    num_envs = env.unwrapped.num_envs

    # record config to wandb
    run.config.update(
        {
            **vars(cfg),
            "task": args_cli.task,
            "num_envs": num_envs,
            "seed": args_cli.seed,
        },
        allow_val_change=True,
    )

    # staggers episode starts, in whole rollouts so envs that only end in truncation
    # don't result in resets every step.
    if cfg.stagger_initial_episodes:
        ep_len_buf = env.unwrapped.episode_length_buf
        rollouts_per_episode = max(
            int(env.unwrapped.max_episode_length) // cfg.num_steps_per_env, 1
        )
        offsets = torch.randint_like(ep_len_buf, high=rollouts_per_episode)
        ep_len_buf.copy_(offsets * cfg.num_steps_per_env)

    steps_per_rollout = cfg.num_steps_per_env * num_envs
    num_steps = cfg.num_steps_per_env

    saver = CheckpointSaver(
        f"ppo_logs/{args_cli.task}/{run_id}/", args_cli.checkpoint_iters
    )

    # resume from a checkpoint. fully resumes the training, with actor, critic,
    # optimizer, lr, and step count for correct curriculum firing.
    start_iter = 0
    if args_cli.resume:
        start_iter = agent.load_checkpoint(args_cli.resume)
        env.unwrapped.common_step_counter = start_iter * cfg.num_steps_per_env
        print(f"Resumed from {args_cli.resume} at iteration {start_iter}.")
    elif cfg.use_normalization:
        # warm-up the observation normalizer. prevents large KL on first rollout.
        state = warmup_normalization(env, agent, state, cfg.num_steps_per_env)

    # per-term reward logging
    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms
    step_dt = env.unwrapped.step_dt

    # tracks episode/term rewards across each rollout, forward-filling reward stats
    # when a rollout has zero completed episodes.
    tracker = EpisodeTracker(num_envs, term_names, device)

    # owns the wandb stream and the live terminal table.
    logger = TrainingLogger(
        args_cli.task, run.url, term_names, cfg.max_iterations, steps_per_rollout
    )
    logger.start()

    # with ctrl+c closing the script, the terminal breaks without this.
    def _on_sigint(signum, frame):
        logger.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    # rollout buffers for one PPO iteration.
    storage = RolloutStorage(
        num_steps, num_envs, state_dim, action_dim, len(term_names), device
    )
    # handles rollout stepping, allows for graph capture for performance.
    rollout = RolloutStepper(agent, storage, step_dt)
    rollout_timer = PhaseTimer(device)
    preparation_timer = PhaseTimer(device)
    update_timer = PhaseTimer(device)

    for update in range(start_iter, cfg.max_iterations):
        rollout_timer.start()
        for step in range(num_steps):
            # select action from policy
            action = rollout.act(policy_obs(state))

            # take step in environment
            state, reward, terminated, truncated, info = env.step(action)

            # store rollout data.
            rollout.record(
                step, reward, terminated, truncated, reward_manager._step_reward
            )

        rollout_timer.stop()
        preparation_timer.start()

        # critic values for the whole rollout in one batched forward pass.
        with torch.no_grad():
            values = agent.evaluate_values(storage.states)
            next_value = agent.evaluate_values(policy_obs(state))

        # accumulate episode/term rewards
        tracker.record_rollout(storage.rewards, storage.dones, storage.term_rewards)

        # compute GAE advantages and returns
        advantages, returns = agent.compute_gae(
            storage.rewards, values, storage.dones, storage.truncs, next_value
        )

        # update actor and critic networks
        preparation_timer.stop()
        update_timer.start()
        batch = storage.flatten(
            returns=returns,
            advantages=advantages,
            values_old=values,
            std_old=rollout.std,
        )
        mean_kl = agent.update(
            batch,
            epochs=cfg.num_learning_epochs,
            num_mini_batches=cfg.num_mini_batches,
        )
        # ensure normalization statistics are fixed for each rollout + update.
        if cfg.use_normalization:
            agent.update_normalization(storage.states.view(-1, state_dim))
        update_timer.stop()

        # reduce rollout into episode statistics (reward stats forward-filled
        # if no episodes completed this rollout; num_episodes is zero then).
        stats = tracker.summarize(agent.entropy())

        logger.log_iteration(
            update,
            stats,
            mean_kl,
            agent.current_lr,
            rollout_timer.seconds(),
            update_timer.seconds(),
            preparation_time=preparation_timer.seconds(),
            warmup=(update == start_iter),
        )

        if args_cli.save:
            saver.save_iteration(
                agent, update + 1, stats.avg_reward, num_episodes=stats.num_episodes
            )

    logger.stop()
    env.close()
    wandb.finish()

    if args_cli.save:
        saver.save_final(agent, cfg.max_iterations)


if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="PPO training for MuJoCo Warp environments."
    )
    parser.add_argument(
        "--num_envs", type=int, default=4096, help="Number of environments to simulate."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to simulate on."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override the config file's max_iterations.",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Enable WandB parameter sweeping."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Enable saving of agents (Policy and Value networks)"
        "every 100 updates, new best performance, and at the end",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint_*.pth bundle to resume training from "
        "(restores weights, optimizer, LR, and iteration).",
    )
    parser.add_argument(
        "--checkpoint_iters",
        type=str,
        default="5000,7500,10000",
        help="Comma-separated iterations at which to save full resumable "
        "checkpoint_<iter>.pth bundles (restore weights, optimizer, LR, and "
        "iteration). Requires --save.",
    )

    # parse the arguments
    args_cli = parser.parse_args()

    try:
        train(args_cli)
    except KeyboardInterrupt:
        wandb.finish(exit_code=255)
        sys.exit(130)
    except BaseException:
        wandb.finish(exit_code=1)
        raise
