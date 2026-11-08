import argparse
import sys

import typing_extensions  # noqa: F401 # for profiling so torch doesnt break
from isaaclab.app import add_launcher_args, launch_simulation


def train(args_cli, overrides):
    args_cli.require_kit = True
    with launch_simulation(None, args_cli):
        _train(args_cli, overrides)


def _train(args_cli, overrides):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.cli import load_external_tasks
    from isaaclab_tasks.utils import parse_env_cfg

    from ryan_ppo.training import train_agent
    from ryan_ppo.utils import Profiler, install_rich_traceback, set_seed

    # registers external tasks for isaaclab.
    load_external_tasks()

    profiler = Profiler(args_cli.profile)
    profiler.begin(1, "train_loop")

    # the suppressed-library list lives in utils.install_rich_traceback.
    install_rich_traceback()

    torch.set_float32_matmul_precision("high")

    set_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        overrides=overrides,
    )
    env_cfg.seed = args_cli.seed

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    train_agent(env, args_cli, "PPO IsaacLab", profiler)

    env.close()

    profiler.end(1)


if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="PPO training for Isaac Lab environments."
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of environments to simulate."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
        "every 100 updates, new best performance, on ctrl+c, and at the end",
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the training loop with carb.profiler zones (Tracy backend).",
    )

    # append the simulation launcher's cli args
    add_launcher_args(parser)

    # parse the arguments
    args_cli, overrides = parser.parse_known_args()

    if args_cli.profile:
        print("Profiling enabled using carb.profiler with Tracy backend.")
        sys.argv.extend(
            [
                "--enable",
                "omni.kit.profiler.tracy",
                "--/profiler/enabled=true",
                "--/app/profilerBackend=tracy",
                "--/privacy/externalBuild=0",
                "--/app/profileFromStart=true",
            ]
        )

    try:
        train(args_cli, overrides)
    except KeyboardInterrupt:
        import wandb

        wandb.finish(exit_code=255)
        sys.exit(130)
    except BaseException:
        import wandb

        wandb.finish(exit_code=1)
        raise
