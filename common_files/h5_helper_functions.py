import h5py
import os
from .variables import REPLAY_MEMORY_SIZE
import numpy as np
#Saving 1 million entries in RAM was too much for my shabby 64GB, so need to save them as files and then have the replay memory store pointers to the files
def save_data_as_h5(game_name, iteration_count, data, is_next_state=False):
    path = 'atari_dqn/env_data/'+game_name+"/"
    os.makedirs(path, exist_ok=True)

    with h5py.File('atari_dqn/env_data/'+game_name+"/" +generate_file_name(iteration_count, is_next_state)+'.h5', 'w') as f:
        #We could just write the dataset as the filename but Im going to set it to be the iteration count for a sanity check
        f.create_dataset(str(iteration_count), data=data)

def load_data_from_h5(game_name, iteration_count, is_next_state=False):
    data = None
    with h5py.File('atari_dqn/env_data/'+game_name+"/"+generate_file_name(iteration_count, is_next_state)+'.h5', 'r') as f:
        try:
            dataset = f[str(iteration_count)]
        #My computer has crashed a few times mid training,
        #This means we can no longer access the data because they have the wrong titles
        #This is kind of like cheating but it allows me to recover after a crash without completly restarting 
        #I dont even know if itll work or if itll completely jank the thing, testing now...
        except KeyError:
            dataset = f[str(iteration_count+1000000)]

        data = np.array(dataset)


    return data

def generate_file_name(iteration_count, is_next_state=False):
    #once we've saved a full replay memory worth of data we want to start overwriting to save disk space
    if is_next_state:
        return str(iteration_count % REPLAY_MEMORY_SIZE)+"_n"
    return str(iteration_count % REPLAY_MEMORY_SIZE)
