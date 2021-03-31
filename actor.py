import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""

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
        
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(400)
        
        # Initialize the hidden layer weights
        self.reset_parameters()

    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3,3e-3)
        self.fc2.weight.data.uniform_(-3e-3,3e-3)
        self.fc3.weight.data.uniform_(-3e-3,3e-3)        

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # Pass the input through all the layers apllying ReLU activation except for the output layer
        x = F.relu(self.fc1(state))
        # Batch Normalization of the first layer
        # x = self.bn(x)
        x = F.relu(self.fc2(x))
        # Pass the result through the output layer apllying tahn function
        x = torch.tanh(self.fc3(x))
        
        return x


