import matplotlib
import matplotlib.pyplot as plt
import torch 

def plot_durations(game_name, episode_durations, is_ipython, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.clf()
    plt.title(game_name)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.full((99,), 0), means))
        plt.plot(means.numpy())

    plt.savefig(game_name+'.png', format='png')