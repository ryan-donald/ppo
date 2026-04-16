[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/whatsnew/3.11.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)
# PPO for IsaacLab
This is a repository containing my implementation of the Proximal Policy Optimizatino (PPO) Reinforcement Learning algorithm, specifically for use in Nvidia's IsaacLab. I initially developed and tested this algorithm within gymnasium, and the base algorithm will still work in those environments, but the *train.py* and *play.py* scripts are specific to IsaacLab, and will need some edits. 

# Quickstart
To use this package, follow the steps below:

* Install and setup Nvidia's IsaacLab, found [here](https://github.com/isaac-sim/IsaacLab).
* Clone this repository.
* Run the command "pip install -e ." within this repository.
* You are all set and can now train agents within IsaacLab using this package. An example training run command, run from the "IsaacLab/" directory, is below:
* "./isaaclab.sh -p -m ryan_ppo.train --task Isaac-Reach-SO-ARM101-Normalized-v0 --num_envs 2048 --headless"

# Features
Fully functional PPO agent, with a configuration file where you can set hyperparameters depending on the task you are running. Additionally, training runs are tracked and stored utilizing Weights and Biases, allowing for easy performance tracking and comparison between runs. 

# Sim2Real
Using this package, I have been able to perform Sim2Real transfer of a Reach agent for the open source SO-ARM101 robot. Specifics about that process can be found [here](https://ryan-donald.github.io/portfolio/1-PPO_Sim2Real/), and my script can be found [here](https://github.com/ryan-donald/so101_ppo).

# Script Structure
The main structure of this repository is as follows:  
* network.py - Contains the *Actor* and *Critic* network classes, used to represent the policy and value functions in the PPO algorithm.  
  
* ppo.py - Contains the *PPOAgent* class, which stores and configures the various hyper-parameters of the algorithm, as well as the following functions:  
  * *select_action* - Selects an action based upon an observation and the current policy.  
  * *compute_gae* - Computes the normalized Generalized Advantage Estimates at each step of the roll-out, as well as the returns.  
  * *update* - Performs the update portion of the PPO algorithm. Given a rollout, this function performs a number of updates, each with a number of mini-batches from the main rollout data.  

* train.py - Contains the training loop, any initialization code, and calls functions to implement the entire PPO algorithm.  
  
* env_cfgs.py - Contains various hyper-parameters for PPO depending on the environment within IsaacLab that is being utilized.
