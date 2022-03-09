from torch import nn
from Models.nn_Model import nn_Model

class MNIST_Model(nn_Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(10 * 6 * 6, 10)
        self.params = nn.ModuleDict({
            'conv1': nn.ModuleList([self.conv1]),
            'classifier': nn.ModuleList([self.out])})

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization
