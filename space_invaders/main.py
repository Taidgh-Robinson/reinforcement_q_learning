import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display
from PIL import Image
import os
import gymnasium as gym
import sys

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
from common_files.model_helper_functions import select_action, optimize_conv_model, push_to_cpu_if_not_None
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

    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(1000000)

    steps_done = 0
    total_frame_count =0 
    episode_durations = []
    episode = 0 

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 1000000
    else:
        num_episodes = 2000

    while total_frame_count < num_episodes:
        # Initialize the environment and get its state
        state = framestack.reset()
        tensor = torch.from_numpy(framestack.get_stack())
        tensor = tensor.to(device)

        for t in count():
            current_state = torch.from_numpy(framestack.get_stack()).float().to(device)
            action = select_action(steps_done, policy_net, framestack.env, current_state)
            steps_done += 1
            observation, reward, terminated, truncated, _ = framestack.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(framestack.get_stack()).float().to(device)

            # Store the transition in memory
            memory.push(push_to_cpu_if_not_None(current_state), push_to_cpu_if_not_None(action), push_to_cpu_if_not_None(next_state), push_to_cpu_if_not_None(reward))
            print(memory)
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_conv_model(memory, policy_net, target_net, optimizer)

            # DQN white paper doesnt use a soft update for the weights 
            # it just copies the weights from the policy net to the target net after 10,000 games have been played
            if(total_frame_count % 10000 == 0):
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                
                target_net.load_state_dict(target_net_state_dict)

            #Save models every 100k frames
            if total_frame_count % 100000 == 0:
                print("GOING TO SAVE MODEL")
                torch.save(target_net.state_dict(), 'space_invaders/models/target_net_' + str(episode) +'.pth')
                torch.save(policy_net.state_dict(), 'space_invaders/models/policy_net_' + str(episode) +'.pth')
            
            total_frame_count+=1 

            if done:
                print("FINISHED EPSISODE: " + str(episode))
                print("Total frame count: " + str(total_frame_count))
                print("Len of replay memory: " + str(len(memory.memory)))
                episode_durations.append(t + 1)
                episode += 1
                break

    torch.save(target_net.state_dict(), 'target_net.pth')
    torch.save(policy_net.state_dict(), 'policy_net.pth')

    print('Complete')
    plot_durations(episode_durations, is_ipython, show_result=False)
    plt.ioff()
    plt.savefig('plot.png', format='png')


#create_gif_from_images("C:\\Users\\taidg\\python\\ML\\DRL\\spaceinvaders\\data", "C:\\Users\\taidg\\python\\ML\\DRL\\spaceinvaders\\data\\gif8.gif", 320)
#run_game_random()
train()
