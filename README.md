# Udcity Deep Reinforcement Learning Project 2: Continuous Control

## Introduction
This project is part of the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), by Udacity.  

The goal of this project is to create and train a double-jointed arm agent that is able to maintain its hand in contact with a moving target for as many time steps as possible.  
![](./images/reacher.gif)


## The environment

This environment has been built using the **Unity Machine Learning Agents Toolkit (ML-Agents)**, which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing this [GitHub repository](https://github.com/Unity-Technologies/ml-agents).  

The project environment provided by Udacity is similar to, but not identical to the Reacher environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).  

In this environment, a double-jointed arm can move to target locations. A reward of **+0.1** is provided for each step that the agent's hand is in the goal location.  
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.


#### State and action spaces
The observation space consists of **33 variables** corresponding to position, rotation, velocity, and angular velocities of the arm.  

Each action is **a vector with four numbers**, corresponding to torque applicable to two joints. Every entry in the action vector should be a number **between -1 and 1**.


#### Solving the environment

There are two versions of the environment.

* **Version 1: One (1) Agent**  
The task is episodic, and in order to solve the environment, the agent must get an **average score of +30 over 100 consecutive episodes**.

* **Version 2: Twenty (20) Agents**  
The barrier to solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
 * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. That yields 20 (potentially different) scores. We then take the average of these 20 scores.  
 * That yields an average score for each episode (where the average is over all 20 agents).  
  
 The environment is considered solved, when the **moving average over 100 episodes** of those average scores **is at least +30**.


## Included in this repository

* Continuous_Control.ipynb - notebook to run the project
* agent.py - ddpg agent implementatioin
* actor.py - actor model implementation
* critic.py - critic model implementation
* checkpoint.pt - saved agent model (actor and critic)
* A Report.md - document describing the solution, the learning algorithm, and ideas for future work
* This README.md file


## Getting Started


### Get the code

##### Option 1. Download it as a zip file

* [Click here](https://github.com/afilimonov/udacity-deeprl-p2-continous-control/archive/master.zip) to download all the content of this repository as a zip file
* Decompress the downloaded file into a folder of your choice

##### Option 2. Clone this repository using Git version control system

```
$ git clone https://github.com/afilimonov/udacity-deeprl-p2-continous-control.git
```

### Install Miniconda

Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib, and a few others.  

If you would like to know more about Anaconda, visit [this link](https://www.anaconda.com/).

In the following links, you find all the information to install **Miniconda** (*recommended*)

* Download the installer: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Installation Guide: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Alternatively, you can install the complete Anaconda Platform

* Download the installer: [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
* Installation Guide: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

### Configure the environment

**1. Create the environment**  
  
```
$ conda create --name udacity-drlnd-p2 python=3.6
$ conda activate udacity-drlnd-p2
```  

**2. Install PyTorch**  
Follow [this link](https://pytorch.org/get-started/locally) to select the right command for your system.  
Here, there are some examples which you can use, if it fit in your system:  

**a.** Linux or Windows

```
## Run this command if you have an NVIDIA graphic card and want to use it  
$ conda install pytorch cudatoolkit=10.1 -c pytorch

## Run this command, otherwise
$ conda install pytorch cpuonly -c pytorch
```  

**b.** Mac  
MacOS Binaries do not support CUDA, install from source if CUDA is needed

```
$ conda install pytorch -c pytorch  
```  

**3. Install Unity Agents**  

```
$ pip install unityagents
```  

### Download the Unity environment with the Agents  

Download the environment from one of the links below and decompress the file into your project folder.  
You need only select the environment that matches your operating system:

 * Version 1: One (1) Agent
     * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
     * Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
     * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
     * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
     * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
 * Version 2: Twenty (20) Agents
     * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
     * Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
     * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
     * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
     * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)     

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining whether your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)


## How to train and test the Agent

Start the Jupyter Notebook using the following commands.

```
$ conda activate udacity udacity-drlnd-p2
$ jupyter notebook
```

To train and test the open `Continuous_Control.ipynb` and execute the cells. The notebook contains additional inforamation on all steps requried to train and test the agent.

#### Additional Information

* [Performance Report](Report.md) - detailed solution walkthrough and performance reports for traing DDPG agent.

