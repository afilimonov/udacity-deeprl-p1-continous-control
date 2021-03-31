import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import time

from actor import Actor
from critic import Critic


_batch_size = 128
_buffer_size = int(1e5)
_gamma = 0.99
_lr_actor = 1e-3
_lr_critic = 1e-3
_tau = 1e-3
_noise_decay = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=_lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=_lr_critic)

        # Initialize target networks weights with the local networks ones
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = _noise_decay

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()

        if noise:
            # Add noise to the action in order to explore the environment
            action += self.noise_decay * self.noise.sample()
            # Decay the noise process along the time
            self.noise_decay *= self.noise_decay

        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        # Save experience
        self.replay_buffer.add(states, actions, rewards, next_states)

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > _batch_size:
            experiences = self.replay_buffer.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        states, actions, rewards, next_states = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions from actor_target model
        actions_next = self.actor_target(next_states)
        # Get predicted next-state Q-Values from critic_target model
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (_gamma * Q_targets_next)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model, tau=_tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, env, n_episodes=2000, max_t=1000, checkpoint_file='checkpoint.pt', target_score = 30.0, mean_window = 100):
        """Deep Deterministic Policy Gradients (DDPG).

        Params
        ======
            env: environment
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            checkpoint_file(str): model file
            target_score(float): target training score
            mean_window(int): window for calculating moving average score 
        """
        scores = []      # episodic scores
        moving_avg = []  # moving average over 100 episodes and over all agents
        
        # Get default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        ## Perform n_episodes of training
        time_training_start = time.time()
        for i_episode in range(1, n_episodes+1):
            time_episode_start = time.time()
            self.noise.reset()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            num_agents = len(env_info.agents)
            scores_episode = np.zeros(num_agents)           # rewards per episode for each agent

            for t in range(1, max_t+1):
                # Perform a step: S;A;R;S'
                actions = self.act(states)                 # select the next action for each agent
                env_info = env.step(actions)[brain_name]    # send the actions to the environment
                rewards = env_info.rewards                  # get the rewards
                next_states = env_info.vector_observations  # get the next states
                # Send the results to the Agent
                for (state, action, reward, next_state) \
                        in zip(states, actions, rewards, next_states):
                    self.step(state, action, reward, next_state)
                # Update the variables for the next iteration
                states = next_states
                scores_episode += rewards

            # Store the rewards and calculate the moving average
            scores.append(scores_episode.tolist())
            moving_avg.append(np.mean(scores[-mean_window:], axis=0))
            # Calculate the elapsed time
            time_episode = time.time() - time_episode_start
            time_elapsed = time.time() - time_training_start
            time_episode_str = time.strftime('%Mm%Ss', time.gmtime(time_episode))

            ## Print the results for this episode
            print('Episode {:3d} ({})\tScore: {:5.2f} \tAverage score: {:5.2f} '
                  .format(i_episode, 
                          time_episode_str, 
                          scores_episode.mean(),
                          moving_avg[-1].mean()))

            ## Check if the environment has been solved
            if moving_avg[-1].mean() >= target_score and i_episode >= mean_window:
                time_elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(time_elapsed))
                print('\nEnvironment solved in {:d} episodes!\t' \
                      'Average Score: {:.2f}\tElapsed time: {}'
                      .format(i_episode-mean_window, moving_avg[-1].mean(), time_elapsed_str))

                ## Save the model 
                self.save(checkpoint_file)

                break

        return scores, moving_avg

    def test(self, env):

        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        
        scores = np.zeros(num_agents)
        
        for i in range(1000):
            actions = self.act(states, noise=False)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards
            scores += rewards

        return scores
    
    def save(self, file):
        """ Save the model """
        checkpoint = {
            'actor_dict': self.actor_local.state_dict(),
            'critic_dict': self.critic_local.state_dict()
            }
        print('\nSaving model ...', end=' ')
        torch.save(checkpoint, 'checkpoint.pt')
        print('done.')
        
    def load(self, file='checkpoint.pt', map_location='cpu'):
        """ Load the trained model """
        checkpoint = torch.load(file, map_location=map_location)
        self.actor_local.load_state_dict(checkpoint['actor_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_dict'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            seed (int): random seed
        """
        self.memory = deque(maxlen=_buffer_size)
        self.experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=_batch_size)
        
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
