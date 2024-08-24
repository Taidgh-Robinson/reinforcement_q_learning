import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
import os

from itertools import count
import torch
import torch.optim as optim

from .models import DQN
from common_files.objects import ReplayMemory
from common_files.plot_helper_functions import plot_durations
from common_files.variables import device, is_ipython, LR, REPLAY_MEMORY_SIZE, K 
from common_files.model_helper_functions import select_action_linearly, optimize_conv_model, preprocess_data_for_memory, save_training_information, clip_reward
from common_files.h5_helper_functions import save_data_as_h5
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

def train(env, episodes_to_train, game_name, p_net = None, t_net = None, mem = None,f_stack = None, t_frame_count = 0, e_data=None):
    assert episodes_to_train % 100000 == 0, "Training steps must be a multiple of 100k, for reasons"
    plt.ion()
    framestack = f_stack if f_stack is not None else FrameStack(env, 4)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    state = framestack.reset()

    policy_net = DQN(n_actions).to(device)
   
    if p_net is not None: 
        policy_net.load_state_dict(p_net)
    target_net = DQN(n_actions).to(device)
    
    if t_net is not None: 
        target_net.load_state_dict(t_net)

    if p_net is None:
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = mem if mem is not None else ReplayMemory(REPLAY_MEMORY_SIZE)

    steps_done = 0
    #Sicne we're resuming we start on the next frame from where we started off, but if we're not resuming we start at 0 
    total_frame_count = t_frame_count if t_frame_count == 0 else t_frame_count + 1
    episode_durations = e_data if e_data is not None else []
    episode = 0 
    num_episodes = total_frame_count + episodes_to_train
    while total_frame_count < num_episodes:
        # Initialize the environment
        framestack.reset()

        for t in count():
            current_state = torch.from_numpy(framestack.get_stack()).float().to(device)

            #we select a new action every k steps, as per paper
            if(t % K == 0):
                action = select_action_linearly(total_frame_count, policy_net, framestack.env, current_state)

            steps_done += 1

            observation, reward, terminated, truncated, _ = framestack.step(action.item())
            reward = torch.tensor([reward], device=device)
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
                optimize_conv_model(game_name, memory, policy_net, target_net, optimizer)

            # DQN white paper doesnt use a soft update for the weights 
            # it just copies the weights from the policy net to the target net after 10,000 games have been played
            if(total_frame_count % 10000 == 0):
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                
                target_net.load_state_dict(target_net_state_dict)

            #Save models every 100k frames
            #Do this because 100k % 4 = 0 and its a reasonable number thats 1/100th of our total
            if total_frame_count % 100000 == 0:
                print("GOING TO SAVE MODEL")
                save_training_information(game_name, total_frame_count, memory, framestack, target_net, policy_net, episode_durations)
            
            total_frame_count+=1 

            if done:
                print("FINISHED EPSISODE: " + str(episode))
                print("CURRENT STEP COUNT: " + str(total_frame_count))
                episode_durations.append(t + 1)
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
    framestack.reset()
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha = 0.95, eps=0.01)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    steps_done = 0
    total_frame_count = 0 
    episode_durations = []
    episode = 0 
    num_episodes = 500000-1
    while total_frame_count < num_episodes:
        # Initialize the environment
        framestack.reset()

        for t in count():
            current_state = torch.from_numpy(framestack.get_stack()).float().to(device)

            #we select a new action every k steps, as per paper
            if(t % K == 0):
                action = select_action_linearly(total_frame_count, policy_net, framestack.env, current_state)

            steps_done += 1

            observation, reward, terminated, truncated, _ = framestack.step(action.item())
            reward = torch.tensor([clip_reward(reward)], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(framestack.get_stack()).float().to(device)

            # tore the transition in memory
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
                optimize_conv_model(game_name, memory, policy_net, target_net, optimizer)

            # DQN white paper doesnt use a soft update for the weights 
            # it just copies the weights from the policy net to the target net after 10,000 frames have been played
            if(total_frame_count+1 % 10000 == 0):
                print("GOING TO UPDATE TARGET NET")
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                
                target_net.load_state_dict(target_net_state_dict)

            #Save models every 100k frames
            #Do this because 100k % 4 = 0 and its a reasonable number thats 1/100th of our total
            if (total_frame_count+1) % 100000 == 0:
                print("GOING TO SAVE MODEL")
                save_training_information(game_name, total_frame_count, memory, framestack, target_net, policy_net, episode_durations)
            
            total_frame_count+=1 

            if done:
                print("FINISHED EPSISODE: " + str(episode))
                print("CURRENT STEP COUNT: " + str(total_frame_count))
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

#env = gym.make("ALE/Breakout-v5")
#train(env, 100000, 'Breakout')

env = gym.make("ALE/Breakout-v5")
start_train(env, "Breakout")