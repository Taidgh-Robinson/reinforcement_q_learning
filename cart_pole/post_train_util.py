import torch
import gym
import warnings
import os
import matplotlib.pyplot as plt

from models import DQN
from variables import device
from PIL import Image
from IPython.display import clear_output
from IPython import display

MODEL_NAME = "policy_net_400.pth"

# Filter out DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

def run_loaded_model_till_failure(env):
    
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    network = DQN(n_observations, n_actions)

    network.load_state_dict(torch.load('models/'+MODEL_NAME, map_location=torch.device('cpu')))
    state = torch.tensor(state, dtype=torch.float32, device=device)

    not_terminated = True
    total_reward = 0
    run_count = 0 
    with torch.no_grad():
        while not_terminated:
            plt.imshow(env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.savefig("data/"+MODEL_NAME+str(run_count)+'.jpg')

            act = network(state)
            max_index = torch.argmax(act).item()

            observation, reward, terminated, truncated, _ = env.step(max_index)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            not_terminated = not terminated and not truncated
            total_reward += reward
            run_count += 1

        print(total_reward)

def create_gif_from_images(image_folder, gif_path, length):    
    image_tags = [i for i in range(length)]
    image_filenames = [MODEL_NAME+str(i)+'.jpg' for i in image_tags]
    # Create a list to store image objects
    images = []
    # Open each image and append it to the list
    for filename in image_filenames:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        images.append(image)
    
    # Save the images as a GIF
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=length//30, loop=0)

run_loaded_model_till_failure()
#create_gif_from_images("data/", "data/300.gif", 159)