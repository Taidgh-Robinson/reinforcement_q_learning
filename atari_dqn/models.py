import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.convLayer1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0)
        self.convLayer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.linearLayer1 = nn.Linear(2592, 256)
        self.linearLayer2 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.convLayer1(x))
        x = F.relu(self.convLayer2(x))
        to_batch_or_not_to_batch = len(x.shape) 
        if(to_batch_or_not_to_batch == 3):
            x = x.view(-1)
        if(to_batch_or_not_to_batch == 4):
            x = x.view(x.size(0), -1)
        x = F.relu(self.linearLayer1(x))
        x = self.linearLayer2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)
