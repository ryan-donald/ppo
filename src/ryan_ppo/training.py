from __future__ import annotations

import signal
from datetime import datetime

import torch
import wandb

from ryan_ppo.checkpointing import CheckpointSaver
from ryan_ppo.config import TrainConfig
from ryan_ppo.ppo import PPOAgent
from ryan_ppo.rollout import RolloutStepper
from ryan_ppo.storage import RolloutStorage
from ryan_ppo.tracking import EpisodeTracker, TrainingLogger
from ryan_ppo.utils import (
    PhaseTimer,
    Profiler,
    env_dims,
    get_cfg_path,
    get_device,
    install_rich_traceback,
    policy_obs,
    warmup_normalization,
)


def train_agent(env, args_cli, wandb_project: str, profiler: Profiler) -> None:
    device = get_device(env.unwrapped.device)

    # get environment-specific training configuration
    cfg = TrainConfig.from_ini(get_cfg_path(args_cli.task))

    if args_cli.max_iterations is not None:
        cfg.max_iterations = args_cli.max_iterations

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(
        project=wandb_project,
        name=f"{args_cli.task}_{run_id}",
        settings={"console": "off"},
    )

    # wandb.init() replaces sys.excepthook; put rich's back or a failure prints
    # the plain traceback first and the rich one second.
    install_rich_traceback()

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
    # optimizer, lr, best reward, and step count for correct curriculum firing.
    start_iter = 0
    if args_cli.resume:
        start_iter, saver.best_reward = agent.load_checkpoint(args_cli.resume)
        env.unwrapped.common_step_counter = start_iter * cfg.num_steps_per_env
        print(f"Resumed from {args_cli.resume} at iteration {start_iter}.")
    elif cfg.use_normalization:
        # warm-up the observation normalizer. prevents large KL on first rollout.
        state = warmup_normalization(env, agent, state, cfg.num_steps_per_env)

    # per-term reward logging
    reward_manager = getattr(env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        # direct
        term_names = []
        step_term_rewards = torch.zeros(num_envs, 0, device=device)
    else:
        # manager based
        term_names = reward_manager.active_terms
        step_term_rewards = reward_manager._step_reward
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
    stop_requested = False

    def _on_sigint(signum, frame):
        nonlocal stop_requested
        logger.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if not args_cli.save:
            raise KeyboardInterrupt
        print("Stopping after this iteration, press ctrl+c again to quit now.")
        stop_requested = True

    signal.signal(signal.SIGINT, _on_sigint)

    # rollout buffers for one PPO iteration.
    storage = RolloutStorage(
        num_steps, num_envs, state_dim, action_dim, len(term_names), device
    )
    # handles rollout stepping, allows for graph capture for performance.
    rollout = RolloutStepper(agent, storage, step_dt)
    batch = storage.flatten(
        returns=storage.returns,
        advantages=storage.advantages,
        values_old=storage.values,
        std_old=rollout.std,
    )
    agent.use_fixed_batch(batch)
    rollout_timer = PhaseTimer(device)
    preparation_timer = PhaseTimer(device)
    update_timer = PhaseTimer(device)

    for update in range(start_iter, cfg.max_iterations):
        rollout_timer.start()
        for step in range(num_steps):
            # select action from policy
            with profiler.zone(2, "select_action"):
                action = rollout.act(policy_obs(state))

            # take step in environment
            with profiler.zone(3, "env_step"):
                state, reward, terminated, truncated, info = env.step(action)

            # store rollout data.
            rollout.record(step, reward, terminated, truncated, step_term_rewards)

        rollout_timer.stop()
        preparation_timer.start()

        # bootstrap value for the state after the last step.
        with torch.no_grad():
            next_value = agent.evaluate_values(policy_obs(state))

        # accumulate episode/term rewards
        tracker.record_rollout(storage.rewards, storage.dones, storage.term_rewards)

        # compute GAE advantages and returns
        with profiler.zone(4, "compute_gae"):
            advantages, returns = agent.compute_gae(
                storage.rewards,
                storage.values,
                storage.dones,
                storage.truncs,
                next_value,
            )
            storage.advantages.copy_(advantages)
            storage.returns.copy_(returns)

        # update actor and critic networks
        preparation_timer.stop()
        update_timer.start()
        with profiler.zone(5, "update"):
            mean_kl = agent.update(
                batch,
                epochs=cfg.num_learning_epochs,
                num_mini_batches=cfg.num_mini_batches,
            )
        # ensure normalization statistics are fixed for each rollout + update.
        if cfg.use_normalization:
            agent.update_normalization(storage.states.view(-1, state_dim))
        update_timer.stop()

        with profiler.zone(6, "logging/cli"):
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

        if stop_requested:
            saver.save_latest(agent, update + 1)
            print(f"Saved {saver.latest_path} at iteration {update + 1}.")
            raise KeyboardInterrupt

    logger.stop()
    wandb.finish()

    if args_cli.save:
        saver.save_final(agent, cfg.max_iterations)
