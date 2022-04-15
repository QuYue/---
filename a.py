#%%
import torch
import torch.nn as nn
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 5, [3,3], padding=2)
                    )
        self.linear = nn.Linear(52020, 2)
        

    def forward(self, X):
        x = self.conv1(X)
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

#%%
if __name__ == "__main__":
    X = torch.ones(7, 3, 100, 100)
    net = Net()

    y = net(X)

    print(y.shape)
    