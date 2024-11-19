import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
import os
import numpy as np
import random
from itertools import count
import torch
import torch.optim as optim
import random

from .models import DQN
from common_files.objects import ReplayMemory
from common_files.plot_helper_functions import plot_durations
from common_files.variables import device, is_ipython, LR, REPLAY_MEMORY_SIZE, K 
from common_files.model_helper_functions import select_action_linearly, optimize_conv_model, preprocess_data_for_memory, save_training_information, clip_reward, load_training_info
from common_files.h5_helper_functions import save_data_as_h5
from common_files.framestack import FrameStack

def continue_training(episodes_to_train, game_name, p_net, t_net, mem,f_stack, t_frame_count, e_data, is_done):
    assert episodes_to_train % 100000 == 0, "Training steps must be a multiple of 100k, for reasons"
    plt.ion()
    framestack = f_stack

    if (is_done):
        f_stack.reset()
    # Get number of actions from gym action space
    n_actions = f_stack.env.action_space.n
    print("NUMBER OF ACTIONS: " + str(n_actions))

    policy_net = DQN(n_actions).to(device)
    policy_net.load_state_dict(p_net)

    target_net = DQN(n_actions).eval().to(device)
    target_net.load_state_dict(t_net)

    #For the first 5 million frames use my juiced learning rate, then use the paper learning rate. This helps us update a working m
    if(t_frame_count >= 4_999_999):
        print("Using paper learning rate")
        optimizer = optim.Adam(params=policy_net.parameters(), lr=5e-5)
    else:
        print("Using juiced learning rate")
        optimizer = optim.Adam(params=policy_net.parameters(), lr=1e-4)

    memory = mem
    current_frames = 0
    steps_done = 0
    total_frame_count = t_frame_count + 1
    episode_durations = e_data if e_data is not None else []
    episode = 0
    num_episodes = total_frame_count + episodes_to_train
    total_reward = 0
    losses = []
    optimizer_count = 0 

    best_model_score = min(max(episode_durations), 17)
    print("BEST MODEL SCORE: ")
    print(str(best_model_score))
    while total_frame_count < num_episodes:
        if episode != 0: 
            framestack.reset()
        for t in count():
            if(total_frame_count > num_episodes): 
                break 
            current_state = torch.from_numpy(framestack.get_stack()).float().to(device)
            #we select a new action every k steps, as per paper
            if(t % K == 0):
                action = select_action_linearly(total_frame_count, policy_net, framestack.env, current_state)

            steps_done += 1

            observation, reward, terminated, truncated, _ = framestack.step(action.item())
            reward = torch.tensor([clip_reward(reward)], dtype=torch.float32, device=device)
            total_reward += reward 
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(framestack.get_stack()).float().to(device)
            
            # Store the transition in memory
            #We write the current state and next state's to files, and then we 
            current_state_processed = preprocess_data_for_memory(current_state)
            save_data_as_h5(game_name, total_frame_count, current_state_processed)

            tmp = None
            if(next_state is not None):
                next_state_processed = preprocess_data_for_memory(next_state)
                save_data_as_h5(game_name, total_frame_count, next_state_processed, True)
                tmp = total_frame_count

            memory.push(total_frame_count, preprocess_data_for_memory(action), tmp, preprocess_data_for_memory(reward))

            # Perform one step of the optimization (on the policy network)
            #We only do this on every kth step as per the paper
            # "More precisely, the agent sees and selects actions on every kth frame instead of every frame, and its last action is repeated on skipped frames"
            if(t % K == 0):
                loss = optimize_conv_model(game_name, memory, policy_net, target_net, optimizer, optimizer_count)
                optimizer_count += 1
                if loss is not None:
                    losses.append(loss)

            # DQN white paper doesnt use a soft update for the weights 
            # it just copies the weights from the policy net to the target net after 10,000 games have been played
            if ((total_frame_count+1) % 10000) == 0:
                print("GOING TO COPY POLICY NET TO TARGET NET ON TFC: " + str(total_frame_count))
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                
                target_net.load_state_dict(target_net_state_dict)

            #Save models every 100k frames
            #Do this because 100k % 4 = 0 and its a reasonable number thats 1/100th of our total
            if (total_frame_count+1) % 100000 == 0:
                print("GOING TO SAVE MODEL")
                save_training_information(game_name, total_frame_count, memory, framestack, target_net, policy_net, episode_durations, done)
            
            total_frame_count+=1 
            current_frames+=1
            if done:
                print("FINISHED EPSISODE: " + str(episode))
                print("TOTAL REWARD: " + str(total_reward))
                print("FRAME COUNT: " + str(current_frames))
                print("CURRENT STEP COUNT: " + str(total_frame_count))
                print("AVERAGE LOSS OVER EPISODES: " + str(sum(losses)/ len(losses)))
                print("MAX LOSS IN EPSIDE: " + str(max(losses)))
                print(" --- ")
                losses = []
                episode_durations.append(total_reward)
                #If this was our best model lets save it to show it off
                if(int(total_reward.cpu().numpy().item()) > best_model_score):
                    print("This was our best model this run with a score of: " + str(int(total_reward.cpu().numpy().item())))
                    path = 'atari_dqn/models/' +game_name+"/"+ str(int(total_reward.cpu().numpy().item()))+'-'+str(total_frame_count) + '/'
                    os.makedirs(path, exist_ok=True)
                    best_model_score = int(total_reward.cpu().numpy().item())
                    torch.save(policy_net.state_dict(), path+'policy_net.pth')

                total_reward = 0
                current_frames = 0
                episode += 1
                break

    torch.save(target_net.state_dict(), 'target_net.pth')
    torch.save(policy_net.state_dict(), 'policy_net.pth')

    print('Complete')
    plot_durations(episode_durations, is_ipython, show_result=False)
    plt.ioff()
    plt.savefig('plot.png', format='png')


#Okay this is wholy unneeded and could just be a part of the resume training method by using a bunch of params with default values of None and
#then doing if param is not none param else : whatever the init value is but that is difficult to read and since I'm presenting this the start training method was born
#Train the model for 100k frames to get everybody cooking
def start_train(env, game_name):
    plt.ion()
    framestack = FrameStack(env, 4)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    print("NUMBER OF ACTIONS: " + str(n_actions))
    framestack.reset()
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).eval().to(device)
    
    optimizer = optim.Adam(params=policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    steps_done = 0
    total_frame_count = 0 
    episode_durations = []
    episode = 0 
    num_episodes = 1_000_000-1
    total_reward = 0
    losses = []
    optimizer_count = 0 
    current_frames = 0
    while total_frame_count < num_episodes:
        # Initialize the environment
        framestack.reset()

        for t in count():
            if(total_frame_count > num_episodes): 
                break 

            current_state = torch.from_numpy(framestack.get_stack()).float().to(device)

            #we select a new action every steps, as per paper
            if(t%4 ==0):
                action = select_action_linearly(total_frame_count, policy_net, framestack.env, current_state)

            steps_done += 1

            observation, reward, terminated, truncated, _ = framestack.step(action.item())
            total_reward += clip_reward(reward)
            reward = torch.tensor([clip_reward(reward)], dtype=torch.float32, device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(framestack.get_stack()).float().to(device)

            #store the transition in memory
            #We write the current state and next state's to files, and then we push pointers to those files to the replay memory
            current_state_processed = preprocess_data_for_memory(current_state)
            save_data_as_h5(game_name, total_frame_count, current_state_processed)

            tmp = None
            if(next_state is not None):
                next_state_processed = preprocess_data_for_memory(next_state)
                save_data_as_h5(game_name, total_frame_count, next_state_processed, True)
                tmp = total_frame_count

            memory.push(total_frame_count, preprocess_data_for_memory(action), tmp, preprocess_data_for_memory(reward))

            # Perform one step of the optimization (on the policy network)
            #We only do this on every kth step as per the paper
            # "More precisely, the agent sees and selects actions on every kth frame instead of every frame, and its last action is repeated on skipped frames"
            if(t % K == 0):
                loss = optimize_conv_model(game_name, memory, policy_net, target_net, optimizer, optimizer_count)
                optimizer_count += 1
                if loss is not None:
                    losses.append(loss)

            # DQN white paper doesnt use a soft update for the weights 
            # it just copies the weights from the policy net to the target net after 10,000 frames have been played
            if ((total_frame_count+1) % 10000) == 0:
                print("GOING TO COPY POLICY NET TO TARGET NET ON TFC: " + str(total_frame_count))
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                
                target_net.load_state_dict(target_net_state_dict)

            #Save models every 100k frames
            #Do this because 100k % 4 = 0 and its a reasonable number thats 1/100th of our total
            if (total_frame_count+1) % 100000 == 0:
                print("GOING TO SAVE MODEL")
                save_training_information(game_name, total_frame_count, memory, framestack, target_net, policy_net, episode_durations, done)
            
            total_frame_count+=1
            current_frames += 1

            if done:
                print("FINISHED EPSISODE: " + str(episode))
                print("TOTAL REWARD: " + str(total_reward))
                print("FRAME COUNT: " + str(current_frames))
                print("CURRENT STEP COUNT: " + str(total_frame_count))
                print("AVERAGE LOSS OVER EPISODES: " + str(sum(losses)/ len(losses)))
                print("MAX LOSS IN EPSIDE: " + str(max(losses)))
                print(" --- ")
                losses = []
                episode_durations.append(total_reward)
                total_reward = 0
                current_frames = 0
                episode += 1
                break

    torch.save(target_net.state_dict(), 'target_net.pth')
    torch.save(policy_net.state_dict(), 'policy_net.pth')

    print('Complete')
    plot_durations(episode_durations, is_ipython, show_result=False)


#create_gif_from_images("C:\\Users\\taidg\\python\\ML\\DRL\\spaceinvaders\\data", "C:\\Users\\taidg\\python\\ML\\DRL\\spaceinvaders\\data\\gif8.gif", 320)
#run_game_random()

#train(env, 100000, 'Breakout')
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")

#env = gym.make("ALE/Breakout-v5")
#start_train(env, "BreakoutD")

data = load_training_info("BreakoutD", 3_999_999)
continue_training(1_000_000, "BreakoutD", data[0], data[1], data[2], data[3], data[4], data[5], data[6])
