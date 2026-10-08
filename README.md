[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/whatsnew/3.12.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-v3.0.0--EA-76B900?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA)
[![Robot](https://img.shields.io/badge/Robot-SO--ARM101-1f6feb?logo=github&logoColor=white)](https://github.com/TheRobotStudio/SO-ARM100)
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
I benchmarked this implementation against the four RL libraries bundled with Isaac Lab: [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [rl_games](https://github.com/Denys88/rl_games), [skrl](https://github.com/Toni-SM/skrl), and [sb3](https://github.com/DLR-RM/stable-baselines3). Every run used 8,192 parallel environments on an RTX 5080. My benchmarks are here: [benchmarks/README.md](benchmarks/README.md).

**Throughput.** My library had the highest training throughput on each task that I tested, when compared to the libraries that ship with Isaac Lab.

| Task | Manager, PhysX | vs next fastest | Direct, Newton | vs next fastest |
|---|---:|---:|---:|---:|
| Cartpole | 1,816,704 | 1.54× | 3,508,178 | 2.02× |
| Ant | 994,079 | 1.11× | 2,097,027 | 1.38× |
| SO-ARM101 Reach | 1,947,558 | 2.06× | 2,880,358 | 2.60× |

Each library has the same physics latency, so the throughput is driven by how fast the surrounding code in a rollout is, and how fast the update step is. I spent time ensuring that there are minimal GPU to host syncs during this, and that the different functions are compiled into efficient CUDA graphs.

**Learning.** To benchmark how well the libraries actually learn (the important part), I used the config for each task that is shipped with Isaac Lab, for each library. For this, two things are important to compare: The reward received by the trained policies, and the time it takes to get there. My library typically performs at least as well as the other libraries in reward gained by the agent, and has the highest throughput.

| Task | `ryan_ppo` final reward | Best other library | `ryan_ppo` wall-clock | Fastest other library |
|---|---:|---:|---:|---:|
| Cartpole | **4.954** | 4.953 (rsl_rl, sb3) | **1.5 min** | 2.1 min (rsl_rl) |
| Ant | **99.0** | 87.6 (rsl_rl) | **9.3 min** | 10.4 min (skrl) |
| Franka Reach | 0.240 | **0.243** (rl_games) | **7.2 min** | 7.9 min (rl_games, skrl) |

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_cartpole_learning.png" width="32%" alt="Cartpole reward vs env steps">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_ant_learning.png" width="32%" alt="Ant reward vs env steps">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_reach_learning.png" width="32%" alt="Franka Reach reward vs env steps">
</div>

In the Ant task, my library learns the best policy of all of them, and has the highest throughput. For some of the other tasks, my library learns a policy atleast as good as the other libraries, with a higher throughput, but for some tasks my library seems to not be as sample efficient early on, so it can occasionally have a slightly shallower curve at the start of training.
