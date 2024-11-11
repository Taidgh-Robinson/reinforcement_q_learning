from PIL import Image
from .framestack import FrameStack
from .models import DQN 
import torch 
import matplotlib.pyplot as plt
from IPython import display
import os 
import gymnasium as gym

def create_gif_from_images(image_folder, gif_path, length,MODEL_NAME):    
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


def run_loaded_model_till_failure(env, game_name, iter_count, MODEL_NAME):

    framestack = FrameStack(env, 4)

    n_actions = env.action_space.n
    # Get the number of state observations
    state = framestack.reset()
    n_observations = len(state)

    network = DQN(n_actions)

    network.load_state_dict(torch.load('atari_dqn/policy_net.pth', map_location=torch.device('cpu')))
    not_terminated = True
    total_reward = 0
    run_count = 0 
    act = None
    k = 4
    with torch.no_grad():
        while not_terminated:
            plt.imshow(framestack.env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.savefig("atari_dqn/data/"+MODEL_NAME+str(run_count)+'.jpg')
            if((run_count % k) == 0):
                simon_says = network(torch.tensor(framestack.get_stack()))
                print(simon_says)
                act = simon_says.max(0).indices.view(1, 1)
            print("SELECTED ACTION: " + action_to_string(act.item()))

            observation, reward, terminated, truncated, _ = framestack.step(act)

            not_terminated = not terminated and not truncated
            total_reward += reward
            run_count += 1

        print(total_reward)

def action_to_string(action):
    action_map = {0 : "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}
    return action_map[action]

#Make a video of an Agent that takes a random action every 4 frames
def make_random_video(): 
    tmp_env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="C:\\Users\\taidg\\python\\ML\\reinforcement_q_learning\\videos", name_prefix="random-agent-Pong")
    observation, info = env.reset()
    env.start_video_recorder()
    is_terminated = False
    step = 0 
    act = None
    while(not is_terminated):
        if((step % 4) == 0):
            act = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(act)
        env.render()
        is_terminated = terminated or truncated

    env.close_video_recorder()
    env.close()


