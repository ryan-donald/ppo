import argparse
import os
from datetime import datetime

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder

from ryan_ppo.config import TrainConfig
from ryan_ppo.ppo import PPOAgent
from ryan_ppo.utils import (
    env_dims,
    get_cfg_path,
    get_device,
    install_rich_traceback,
    policy_obs,
    set_seed,
)


def play(args_cli):
    # the suppressed-library list lives in utils.install_rich_traceback.
    install_rich_traceback()

    set_seed(args_cli.seed)

    env_cfg = load_env_cfg(args_cli.task, play=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # create environment. rgb_array rendering is only needed when recording.
    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRlEnv(env_cfg, device=args_cli.device, render_mode=render_mode)
    device = get_device(env.unwrapped.device)

    if args_cli.video:
        video_dir = os.path.join(
            "logs",
            "test_videos",
            args_cli.task,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"Recording video to {video_dir}")

    # the task's config gives the network shape the checkpoint was trained with.
    cfg = TrainConfig.from_ini(get_cfg_path(args_cli.task))

    state_dim, action_dim = env_dims(env)

    agent = PPOAgent(state_dim, action_dim, cfg, device=device)
    agent.actor.eval()

    # accepts either a bare actor state_dict or a full checkpoint bundle.
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        checkpoint = checkpoint["actor"]
    agent.actor.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {args_cli.checkpoint}")

    state, info = env.reset()
    print(f"Playing {args_cli.eval_steps} steps with {env.unwrapped.num_envs} envs.\n")

    for _ in range(args_cli.eval_steps):
        # handle both Dict and Box observation spaces
        state_obs = policy_obs(state)

        # deterministic action: the policy mean, with no exploration noise.
        with torch.no_grad():
            mu = agent.act_deterministic(state_obs)

        state, reward, terminated, truncated, info = env.step(mu)

    # closing flushes the final video file.
    env.close()


if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="Deterministic playback of a trained policy for MuJoCo Warp tasks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file for actor network.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=50, help="Number of environments to simulate."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to simulate on."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=2000,
        help="Number of steps to run the policy for.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record video of the run.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=500,
        help="Length of recorded video (in steps).",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Interval between videos (in steps).",
    )

    # parse the arguments
    args_cli = parser.parse_args()

    play(args_cli)
