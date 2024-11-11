import math
import random
import torch.nn as nn

from .h5_helper_functions import load_data_from_h5
from .variables import EPS_DECAY, EPS_START, EPS_END, device, BATCH_SIZE, GAMMA, REPLAY_MEMORY_SIZE, EPS_END_STEP_COUNT
from .objects import Transition
from torch import from_numpy
import os
import torch
import pickle

def preprocess_data_for_memory(something):
    if something is not None:
        something = something.cpu()
        return something.numpy()
    return something

def save_training_information(game_name, iteration_number, memory, frameStack, target_net, policy_network, episode_durations, is_done):
    path = 'atari_dqn/models/' +game_name+"/"+ str(iteration_number) + '/'
    os.makedirs(path, exist_ok=True)
    torch.save(policy_network.state_dict(), path+'policy_net.pth')
    torch.save(target_net.state_dict(), path+'target_net.pth')

    with open(path+'memory.pkl', 'wb') as f:
        pickle.dump(memory, f)
    with open(path+'framestack.pkl', 'wb') as f:
        pickle.dump(frameStack, f)
    with open(path+'count.pkl', 'wb') as f:
        pickle.dump(iteration_number, f)
    with open(path+'episode_durations.pkl', 'wb') as f:
        pickle.dump(episode_durations, f)
    with open(path+'is_done.pkl', 'wb') as f:
        pickle.dump(is_done, f)

def load_training_info(game_name, iteration_number): 
    path = 'atari_dqn/models/' + game_name + "/" + str(iteration_number) + '/'
    p_net = torch.load(path+'policy_net.pth')
    t_net = torch.load(path+'target_net.pth')
    
    with open(path+'memory.pkl', 'rb') as file:
        memory_data = pickle.load(file)
    with open(path+'framestack.pkl', 'rb') as file:
        framestack_data = pickle.load(file)
    with open(path+'count.pkl', 'rb') as file:
        count_data = pickle.load(file)
    with open(path+'episode_durations.pkl', 'rb') as file:
        episode_data = pickle.load(file)
    with open(path+'is_done.pkl', 'rb') as file:
        is_done = pickle.load(file)

    return [p_net, t_net, memory_data, framestack_data, count_data, episode_data, is_done]
    

def clip_reward(reward):
    if reward > 0:
        return 1 
    if reward == 0: 
        return 0
    if reward < 0: 
        return -1

#Epsilon Greedy Selection
def select_action(steps_done, policy_net, env, state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max1 result is index of where max element was
            # found, so we pick action with the larger expected reward.
            simon_says = policy_net(state)
            if(len(simon_says.shape) == 1):
                return simon_says.max(0).indices.view(1, 1)
            if(len(simon_says.shape) == 2):
                return simon_says.max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


#Epsilon Greedy Selection with linear annelment (whatever the fuck that means)
def select_action_linearly(steps_done, policy_net, env, state):
    sample = random.random()
    if(steps_done <= EPS_END_STEP_COUNT):
        eps_threshold = EPS_START + ((steps_done / (EPS_END_STEP_COUNT - 1)) * (EPS_END - EPS_START))
    else: 
        eps_threshold = EPS_END

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max1 result is index of where max element was
            # found, so we pick action with the larger expected reward.
            simon_says = policy_net(state)
            if(len(simon_says.shape) == 1):
                return simon_says.max(0).indices.view(1, 1)
            if(len(simon_says.shape) == 2):
                return simon_says.max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):

    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_batch = state_batch.to(device=device)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def optimize_conv_model(game_name, memory, policy_net, target_net, optimizer, optimizer_count):
    
    if len(memory) < BATCH_SIZE:
        print("not enough data in memory, passing")
        return

    transitions = memory.sample(BATCH_SIZE)
    tranistions2 = []
    for t in transitions:
        current_state = load_data_from_h5(game_name, t.state)
        current_state = current_state
        next_state = None
        if(t.next_state is not None):
            next_state = load_data_from_h5(game_name, t.next_state, True)
            next_state = next_state
        reshaped_transition = Transition(current_state, t.action, next_state, t.reward)

        tranistions2.append(reshaped_transition)

    batch = Transition(*zip(*tranistions2))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.stack([from_numpy(s) for s in batch.next_state
                                                if s is not None]).to(device)
    
    state_batch = torch.stack([from_numpy(s) for s in batch.state]).to(device)
    action_batch = torch.stack([from_numpy(a) for a in batch.action]).to(device)
    reward_batch = torch.stack([from_numpy(r) for r in batch.reward]).to(device)
        
    state_action_values = policy_net(state_batch).gather(1, action_batch.squeeze(1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze(1)

    if(((optimizer_count % 100000) == 0)):
        print(' --- ')
        print("action batch:")
        print(action_batch.squeeze(1))
        print("reward batch:")
        print(reward_batch.squeeze(1))
        print("state_action_values: ")
        print(state_action_values)
        print("next_state_values: ")
        print(next_state_values)
        print("expected_state_action_values.unsqueeze(1): ")
        print(expected_state_action_values)
        print(' --- ')

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()

    loss.backward()    
    optimizer.step()
    return loss.item()