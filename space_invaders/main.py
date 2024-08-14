import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display
from PIL import Image
import os
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
from common_files.image_helper_functions import preprocess_image
from common_files.framestack import FrameStack

def run_game_random():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    env.reset() 
    DONE = False
    step = 0 
    while not DONE:
        env.action_space.sample()
        plt.imshow(env.render())
        display.display(plt.gcf())
        display.clear_output(wait=True)
        plt.savefig("data/" + str(step)+'.jpg')
        state, reward, done, info, _ = env.step(env.action_space.sample())
        DONE = done
        print(str(step) + " REWARD: " + str(reward))
        step += 1


def create_gif_from_images(image_folder, gif_path, length):    
    image_tags = [i for i in range(length)]
    image_filenames = [str(i)+'.jpg' for i in image_tags]
    # Create a list to store image objects
    images = []
    # Open each image and append it to the list
    for filename in image_filenames:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        images.append(image)
    
    # Save the images as a GIF
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=length//8, loop=0)

def train():
    env = gym.make("ALE/SpaceInvaders-v5")
    plt.ion()
    framestack = FrameStack(env, 4)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = framestack.reset()
    n_observations = len(state)
    print("n_observations")
    print(n_observations)

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
        state = framestack.reset()

        state = torch.tensor(state, dtype=torch.float32, device=device).view(3, 210, 160)
        print("HERE TAIDGH")
        print(state.shape)
        tensor = torch.from_numpy(framestack.get_stack())
        tensor = tensor.to(device)
        print(tensor.shape)

        print(policy_net(tensor))

        for t in count():
            action = select_action(steps_done, policy_net, framestack.env, framestack.get_stack())
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
                torch.save(target_net.state_dict(), 'space_invaders/models/target_net_' + str(i_episode) +'.pth')
                torch.save(policy_net.state_dict(), 'space_invaders/models/policy_net_' + str(i_episode) +'.pth')

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

#create_gif_from_images("C:\\Users\\taidg\\python\\ML\\DRL\\spaceinvaders\\data", "C:\\Users\\taidg\\python\\ML\\DRL\\spaceinvaders\\data\\gif8.gif", 320)
#run_game_random()
train()
