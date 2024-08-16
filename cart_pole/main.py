import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .models import DQN
from common_files.objects import ReplayMemory, Transition
from common_files.plot_helper_functions import plot_durations
from common_files.variables import device, is_ipython, TAU, LR
from common_files.model_helper_functions import select_action, optimize_model

env = gym.make("CartPole-v1")
plt.ion()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 550

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(steps_done, policy_net, env, state)
        steps_done += 1
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(memory, policy_net, target_net, optimizer)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if i_episode % 10 == 0:
            torch.save(target_net.state_dict(), 'cart_pole/models/target_net_' + str(i_episode) +'.pth')
            torch.save(policy_net.state_dict(), 'cart_pole/models/policy_net_' + str(i_episode) +'.pth')

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations, is_ipython)
            break


torch.save(target_net.state_dict(), 'target_net.pth')
torch.save(policy_net.state_dict(), 'policy_net.pth')

print('Complete')
plot_durations(episode_durations, is_ipython, show_result=True)
plt.ioff()
plt.show()