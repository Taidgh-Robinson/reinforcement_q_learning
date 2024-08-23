import h5py
from .variables import REPLAY_MEMORY_SIZE
import numpy as np
#Saving 1 million entries in RAM was too much for my shabby 64GB, so need to save them as files and then have the replay memory store pointers to the files
def save_data_as_h5(iteration_count, data, is_next_state=False):
    with h5py.File('atari_dqn/env_data/'+generate_file_name(iteration_count, is_next_state)+'.h5', 'w') as f:
        #We could just write the dataset as the filename but Im going to set it to be the iteration count for a sanity check
        f.create_dataset(str(iteration_count), data=data)

def load_data_from_h5(iteration_count, is_next_state=False):
    data = None
    with h5py.File('atari_dqn/env_data/'+generate_file_name(iteration_count, is_next_state)+'.h5', 'r') as f:
        dataset = f[str(iteration_count)]
        data = np.array(dataset)

    return data

def generate_file_name(iteration_count, is_next_state=False):
    #once we've saved a full replay memory worth of data we want to start overwriting to save disk space
    if is_next_state:
        return str(iteration_count % REPLAY_MEMORY_SIZE)+"_n"
    return str(iteration_count % REPLAY_MEMORY_SIZE)
