import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.convLayer = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.layer5 = nn.Linear(6889, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.convLayer(x)
        # Flatten the output of Conv2d layer
        flat = x.view(x.size(0), -1)
        print("In model Taidgh")
        print(flat)
        print(flat.shape)

 
        x = F.relu(self.layer5(flat))

        return x