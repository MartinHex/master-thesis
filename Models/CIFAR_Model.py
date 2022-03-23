from torch import nn
from torchvision.models import resnet
from Models.nn_Model import nn_Model

class CIFAR_Model(nn_Model):
    def __init__(self):
        super().__init__()
        number_of_classes = 100
        self.model = resnet.resnet18(pretrained = False) #, norm_layer=MyGroupNorm)
        self.model.fc = nn.Linear(512, number_of_classes)


    def forward(self, x):
        output = self.model(x)
        return output, x    # return x for visualization

class MyGroupNorm(nn.Module):
    # epsilon = 0.001 seem to be default in FedPA implementation
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-3, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x
