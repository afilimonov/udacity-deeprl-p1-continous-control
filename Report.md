### Solution Walkthrough


#### The environment

In this environment, a double-jointed arm can move to target locations. A reward of **+0.1** is provided for each step that the agent's hand is in the goal location.  
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

![](./images/reacher.gif)

The observation space consists of **33 variables** corresponding to position, rotation, velocity, and angular velocities of the arm.  

Each action is **a vector with four numbers**, corresponding to torque applicable to two joints. Every entry in the action vector should be a number **between -1 and 1**.


#### Solving the environment

This implementation is solving Verson 2 of the environment with 20 agents.


#### Algorithm

The alorithm used in this implementaton is [Deep Deterministic Policy Gradient algorithm DDPG](https://arxiv.org/abs/1509.02971)

