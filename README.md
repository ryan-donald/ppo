[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/whatsnew/3.12.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-v3.0.0--EA-76B900?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA)
![Tests](https://github.com/ryan-donald/ppo/actions/workflows/tests.yaml/badge.svg)

# PPO for IsaacLab
This is a repository containing my implementation of the Proximal Policy Optimization (PPO) Reinforcement Learning algorithm, specifically for use in Nvidia's IsaacLab. I initially developed and tested this algorithm within gymnasium, and then moved to IsaacLab. The base algorithm is not specific to the environment, and will work with any environment as long as the batch data is in the expected format.

<img width="720" height="405" alt="so101_reach" src="https://github.com/user-attachments/assets/a04e37d7-f6f0-4f09-af24-27157c920124" />

# Quickstart
To use this package, follow the steps below:

* Install and setup Nvidia's IsaacLab, found [here](https://github.com/isaac-sim/IsaacLab).
* Install my custom IsaacLab tasks from [tasks-isaaclab](https://github.com/ryan-donald/tasks-isaaclab), which provides the `ryan_tasks` package the training scripts import: clone it and run "pip install -e source/ryan_tasks" in its base directory.
* Clone this repository.
* Run the command "pip install -e ." within this repository.
* You are all set and can now train agents within IsaacLab using this package. An example training run command is below:
* "python -m ryan_ppo.isaaclab.train --task Ryan-Reach-SO-ARM101-Normalized-v0 --num_envs 2048 --headless".

# Features
Fully functional PPO agent, with a configuration file where you can set hyperparameters depending on the task you are running. Additionally, training runs are tracked and stored utilizing Weights and Biases, allowing for easy performance tracking and comparison between runs. 

## Multiple Environments
The base algorithm, defined in the files within the 'src/' directory, are portable to any gym-style environment. Within the 'src/' directory is an 'isaaclab/' directory containing a *train.py* and *play.py* file, which implement the algorithm specifically for IsaacLab. To use the algorithm in another set of environments, simply create your own *train.py* and *play.py* files for those environments in this format. 

## Weights and Biases Parameter Sweeping
This implementation supports parameter sweeping via Weights and Biases. To do this, create a YAML description file in the format of those in "cfg/sweeps/". Within this file, define either a set of discrete values or a distribution for each parameter that you want to be swept. Ensure that *train.py* contains checks for all of the parameters that are being swept to ensure they are actually being used in the runs. After this, run "wandb sweep <sweep config file>" followed by "wandb agent <username>/<project name>/<sweep id>". The results will be logged via Weights and Biases. Shown below is an example plot showing 50 different runs with a reach task, sweeping over a handful of parameters.

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/so101_reach_sweep.png" width="100%" alt="Parameter Sweep">
</div>

## Sim2Real
Using this package, I have been able to perform Sim2Real transfer of a Reach agent for the open source SO-ARM101 robot. Specifics about that process can be found [here](https://ryan-donald.github.io/portfolio/1-PPO_Sim2Real/), and my script can be found [here](https://github.com/ryan-donald/so101_ppo).

[![PPO SO-ARM101 sim2real](https://img.youtube.com/vi/MzxyW7mrM0s/maxresdefault.jpg)](https://www.youtube.com/watch?v=MzxyW7mrM0s)

## Experiment Tracking in Terminal
With the help of the python package [rich](https://github.com/textualize/rich), I have a display in the terminal which provides information about the currently running experiment, including reward terms, learning parameters, performance, remaining time, and a clickable link to the current WandB run. An example of this can be seen below:

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/terminal_display.png" width="100%" alt="Parameter Sweep">
</div>

## Training Run Profiling with Tracy Profiler
Based on the recommendation in the official Isaac Lab documentation [here](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/utilities/debugging/profiling_performance.html), I added support for code profiling using the Tracy profiler. This allows for live profiling of the performance of the training script. This will provide information for the time spent in each block of execution provide information that can be used to gauge and improve the efficiency of the training script, and various environments. To use this, simply add these flags to the script: "--profile --enable omni.kit.profiler.tracy".

# Benchmarks
I benchmarked this implementation against the four RL libraries bundled with Isaac Lab — [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [rl_games](https://github.com/Denys88/rl_games), [skrl](https://github.com/Toni-SM/skrl), and [sb3](https://github.com/DLR-RM/stable-baselines3) on three tasks, cartpole, ant, and my SO-ARM101 reach task. Every run used 12,288 parallel environments, headless, and each library's agent config matched. This is a throughput speed measurement, however my library has a slightly longer startup due to pytorch compilation of some functions. In a short run like cartpole, this could effect the total time more significantly than longer runs. I think with the speedup that they provide, especially for more complex tasks that require long runs, the benefits outweigh this penalty.

**Cartpole** — 16 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,364,807** | **144** | **21** | — |
| skrl | 1,062,592 | 185 | 45 | 1.28× |
| rl_games | 1,046,086 | 188 | 46 | 1.30× |
| rsl_rl | 1,013,389 | 194 | 49 | 1.35× |
| sb3 | 615,140 | 320 | — | 2.22× |

**Ant** — 32 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **632,373** | **622** | **131** | — |
| rl_games | 574,987 | 684 | 186 | 1.10× |
| rsl_rl | 571,742 | 688 | 189 | 1.11× |
| skrl | 570,402 | 689 | 196 | 1.11× |
| sb3 | 388,650 | 1,012 | — | 1.63× |

**Reach** — 24 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,323,175** | **223** | **60** | — |
| skrl | 961,209 | 307 | 135 | 1.38× |
| rl_games | 935,929 | 315 | 129 | 1.41× |
| rsl_rl | 741,893 | 398 | 152 | 1.78× |
| sb3 | 489,900 | 602 | — | 2.70× |

`ryan_ppo` has the highest throughput on every task: 1.10–1.38× the next-fastest library and 1.6–2.7× sb3. For every task, the execution time for the physics is largely unchanged library to library, and the library implementation controls the interface of the agent with the physics, and the update steps of the PPO algorithm. My library has a sigificantly faster update portion, and some of the surrounding framework for the rollouts is also optimized better.

### Faster environments

Each of the three tasks also has a direct workflow version running on Newton (MuJoCo Warp) physics with a CUDA-graph-captured physics step, instead of going through Isaac Lab's managers on PhysX. In these tasks, both the usage of a direct workflow instead of a manager based workflow, and the usage of the Newton physics backend instead of the PhysX backend improve the throughput of the task on the environment side. Same benchmark:

**Cartpole (direct, Newton)** — 16 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **2,802,901** | **70** | **21** | — |
| skrl | 1,886,143 | 104 | 45 | 1.49× |
| rsl_rl | 1,748,061 | 112 | 51 | 1.60× |
| rl_games | 1,747,706 | 112 | 49 | 1.60× |
| sb3 | 815,946 | 241 | — | 3.44× |

**Ant (direct, Newton)** — 32 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **912,864** | **431** | **128** | — |
| rsl_rl | 767,357 | 512 | 181 | 1.19× |
| rl_games | 747,892 | 526 | 188 | 1.22× |
| skrl | 745,901 | 527 | 193 | 1.22× |
| sb3 | 516,538 | 761 | — | 1.77× |

**Reach (direct, Newton)** — 24 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,510,192** | **195** | **59** | — |
| skrl | 1,118,606 | 264 | 128 | 1.35× |
| rl_games | 1,066,240 | 277 | 121 | 1.42× |
| rsl_rl | 695,369 | 424 | 151 | 2.17× |
| sb3 | 564,519 | 522 | — | 2.68× |