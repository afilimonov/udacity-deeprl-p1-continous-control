import torch
import torch.nn as nn
import torch.nn.functional as F
from actor import Actor


class Critic(Actor):
    """Critic (Value) Model."""

    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden (int): Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(128)
        
        # Initialize the hidden layer weights
        self.reset_parameters()

    
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Pass the states into the first layer
        # Pass the input through all the layers apllying ReLU activation except for the output layer
        x = F.relu(self.fc1(state))
        # Batch Normalization of the first layer
        x = self.bn(x)
        # Concatenate the first layer output with the action
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        # Pass the input through all the layers apllying ReLU activation, but the last
        x = torch.sigmoid(self.fc3(x))
        # Return the Q-Value for the input state-action
        return x
