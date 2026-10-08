# Benchmark results

## Throughput

I benchmarked this implementation against the four RL libraries bundled with Isaac Lab — [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [rl_games](https://github.com/Denys88/rl_games), [skrl](https://github.com/Toni-SM/skrl), and [sb3](https://github.com/DLR-RM/stable-baselines3) on three tasks, cartpole, ant, and my SO-ARM101 reach task. Every run used 8,192 parallel environments on an RTX 5080, headless, and each library's agent config matched. This is a throughput speed measurement, however my library has a slightly longer startup due to pytorch compilation of some functions. In a short run like cartpole, this could affect the total time more significantly than longer runs. I think with the speedup that they provide, especially for more complex tasks that require long runs, the benefits outweigh this penalty.

**Cartpole** — 16 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,816,704** | **72** | **5** | — |
| skrl | 1,183,419 | 111 | 34 | 1.54× |
| rl_games | 1,173,036 | 112 | 31 | 1.55× |
| rsl_rl | 1,136,048 | 115 | 35 | 1.60× |
| sb3 | 694,571 | 189 | — | 2.62× |

**Ant** — 32 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **994,079** | **264** | **27** | — |
| skrl | 893,478 | 293 | 54 | 1.11× |
| rl_games | 885,594 | 296 | 49 | 1.12× |
| rsl_rl | 832,319 | 315 | 56 | 1.19× |
| sb3 | 508,995 | 515 | — | 1.95× |

**Reach** — 24 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,947,558** | **101** | **16** | — |
| skrl | 946,503 | 208 | 119 | 2.06× |
| rl_games | 880,621 | 223 | 127 | 2.21× |
| rsl_rl | 657,378 | 299 | 139 | 2.96× |
| sb3 | 546,775 | 360 | — | 3.56× |

`ryan_ppo` has the highest throughput on every task: 1.11–2.06× the next-fastest library and 2.0–3.6× sb3. For every task, the execution time for the physics is largely unchanged library to library, and the library implementation controls the interface of the agent with the physics, and the update steps of the PPO algorithm. My library has a significantly faster update portion, and some of the surrounding framework for the rollouts is also optimized better.

## Faster environments

Each of the three tasks also has a direct workflow version running on Newton (MuJoCo Warp) physics with a CUDA-graph-captured physics step, instead of going through Isaac Lab's managers on PhysX. In these tasks, both the usage of a direct workflow instead of a manager based workflow, and the usage of the Newton physics backend instead of the PhysX backend improve the throughput of the task on the environment side. Same benchmark:

**Cartpole (direct, Newton)** — 16 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **3,508,178** | **37** | **4** | — |
| skrl | 1,739,236 | 75 | 34 | 2.02× |
| rl_games | 1,654,663 | 79 | 34 | 2.12× |
| rsl_rl | 1,636,259 | 80 | 36 | 2.14× |
| sb3 | 845,351 | 155 | — | 4.15× |

**Ant (direct, Newton)** — 32 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **2,097,027** | **125** | **27** | — |
| skrl | 1,522,399 | 172 | 54 | 1.38× |
| rl_games | 1,494,115 | 175 | 50 | 1.40× |
| rsl_rl | 1,325,754 | 198 | 57 | 1.58× |
| sb3 | 648,647 | 404 | — | 3.23× |

**Reach (direct, Newton)** — 24 steps/env

| Framework | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **2,880,358** | **68** | **16** | — |
| skrl | 1,106,186 | 178 | 118 | 2.60× |
| rl_games | 1,062,657 | 185 | 126 | 2.71× |
| sb3 | 620,027 | 317 | — | 4.65× |
| rsl_rl | 503,670 | 390 | 145 | 5.72× |

## Learning

The above sections focus on how the training throughput of my library compares to the other libraries, measured with short runs that don't learn a full policy. This is only relevant if my library also learns the tasks as well as them. To measure this, I also ran a number of runs to verify the learned behavior is on par or better than the other libraries. I did this with three tasks, Cartpole, Ant, and Franka Reach. Each library performs a training run on the tasks, one for each of three set seeds (42, 43, 44), using the config that is provided with Isaac Lab by that library for each task. These were all run on an RTX 5080, with 8192 parallel envs. Some adjustments had to be made for rl_games and sb3, to scale from the default 4096 envs to 8192 envs. The rewards are the median across the three seeds, and the "final" rewards value is the mean of the last 10% of the training run. Range is 25th-75th percentile.

**Cartpole** — 131M env steps (1,000 iterations)

| Framework | Final reward | Seed range | Throughput (steps/s) | Wall-clock (min) |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **4.954** | 4.951–4.954 | **1,820,493** | **1.5** |
| rsl_rl | 4.953 | 4.940–4.954 | 1,172,452 | 2.1 |
| sb3 | 4.953 | 4.951–4.956 | 193,833 | 11.5 |
| skrl | 4.922 | 4.888–4.927 | 837,575 | 2.9 |
| rl_games | 4.797 | 4.782–4.801 | 784,086 | 3.0 |

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_cartpole_learning.png" width="100%" alt="Cartpole reward vs env steps">
</div>

**Ant** — 524M env steps (2,000 iterations at 32 steps/env; 4,000 for the libraries that ship 16 steps/env)

| Framework | Final reward | Seed range | Throughput (steps/s) | Wall-clock (min) |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **99.0** | 97.1–99.2 | **990,794** | **9.3** |
| rsl_rl | 87.6 | 87.5–88.0 | 667,776 | 13.5 |
| skrl | 68.9 | 67.3–74.0 | 877,204 | 10.4 |
| rl_games | 64.7 | 62.5–68.9 | 861,793 | 10.6 |
| sb3 | 59.3 | 58.0–61.1 | 561,684 | 16.0 |

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_ant_learning.png" width="100%" alt="Ant reward vs env steps">
</div>

**Franka Reach** — 197M env steps (1,000 iterations)

| Framework | Final reward | Seed range | Throughput (steps/s) | Wall-clock (min) |
|---|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | 0.240 | 0.239–0.240 | **503,474** | **7.2** |
| rl_games | **0.243** | 0.243–0.243 | 452,301 | 7.9 |
| skrl | 0.242 | 0.242–0.243 | 449,226 | 7.9 |
| rsl_rl | 0.237 | 0.236–0.237 | 441,120 | 8.1 |

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_reach_learning.png" width="100%" alt="Franka Reach reward vs env steps">
</div>

In the Ant task, my library clearly outperforms the others in this small benchmark, in both learning and speed. For the other two tasks, all the libraries learn similar policies, however mine finishes training first. For these tasks, my library appears to have some sample inefficiency early on compared to the others, so it doesn't necessarily reach every reward threshold before the others. The takeaway I had from this is that my library is able to accurately learn tasks comparatively to the shipped libraries, while having a significantly higher training throughput due to the effort spent on optimizing portions of the loop.