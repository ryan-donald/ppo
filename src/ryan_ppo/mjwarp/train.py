import argparse
import sys

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from ryan_ppo.training import train_agent
from ryan_ppo.utils import Profiler, install_rich_traceback, set_seed


def train(args_cli):
    # the suppressed-library list lives in utils.install_rich_traceback.
    install_rich_traceback()

    torch.set_float32_matmul_precision("high")

    set_seed(args_cli.seed)

    env_cfg = load_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # create environment
    env = ManagerBasedRlEnv(env_cfg, device=args_cli.device)

    # carb profiling is isaaclab-only, so the zones are no-ops here.
    train_agent(env, args_cli, "PPO MuJoCo Warp", Profiler(enabled=False))

    env.close()


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
