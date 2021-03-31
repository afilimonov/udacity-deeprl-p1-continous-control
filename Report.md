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

The alorithm used in this implementaton is [Deep Deterministic Policy Gradient algorithm DDPG](https://arxiv.org/abs/1509.02971). DDPG is based on two neural networks Actor and Critic with the actor used to estimate the best action and the critict then used to evaluate the optimal actioin value function.

The Actor Network receives as input 33 variables representing the observation space and generates as output 4 numbers representing the predicted best action for that observed state. That means, the Actor is used to approximate the optimal policy _π_ deterministically.

The Critic Network receives as input 33 variables representing the observation space. The result of the first hidden layer and the action proceeding from the Actor Network are combined as input for the second hidden layer. The output of this network is the prediction of the target value based on the given state and the estimated best action. In other words the critic calculates the optimal action-value function _Q(s, a)_ based on the Actor best estimated action.

#### Network architecture

The fineal network architectures for Actor and Critic include.

_Actor_ 
* First fully connected layer with input size 33 and output size 128
* Second fully connected layer with input size 128 and output size 128
* Third fully connected layer with input size 128 and output size 4
* Cirtic has Batch Normalization layer between first and second layers with input size 128

_Critic_ 
* First fully connected layer with input size 33 and output size 128
* Second fully connected layer with input size 133 and output size 128
* Third fully connected layer with input size 128 and output size 1
* Batch Normalization layer between first and second layers with input size 128

I started with 2 hidden layers sizes 400 and 300 for both actor and crtic as suggested in the DDPG paper and also tried [512, 384], [384, 256] and end up with [128, 128] model that converged well with stable training peformance. I also tried using Batch Normalization for the actor network but it resulted in singificatly higer score fluctuation during the training.




