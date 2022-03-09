from torch import nn
from Models.nn_Model import nn_Model

class EMNIST_Model(nn_Model):
    """CNN model indended to be used on the EMNIST-62 dataset

    This model is a replica of the model used by S. Reddit et al. (https://arxiv.org/pdf/2003.00295.pdf) and by M. AlShedivat et al. (https://arxiv.org/pdf/2010.05273.pdf)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 0,
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 0,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
            padding = 0,
        )
        self.dropout1 = nn.Dropout(p = 0.25)
        self.dropout2 = nn.Dropout(p = 0.5)
        self.dense1 = nn.Linear(9216, 128)
        self.dense2 = nn.Linear(128, 62)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        output = self.softmax(x)
        return output, x    # return x for visualization
