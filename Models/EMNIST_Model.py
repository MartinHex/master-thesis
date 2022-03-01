from torch import nn
from torch.autograd import Variable
import copy

class EMNIST_Model(nn.Module):
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

    def predict(self, input, device = None):
        self.eval()
        if (device!= None): self.to(device)
        if (device != None): input = input.to(device)
        return self(input)

    def evaluate(self,dataloader,loss_func, device = None):
        """Evaluates the model on a given loss function.

            Arguments:
                - dataloader: Dataloader with data to evaluate on.
                - loss_func: Loss function to use for evaluation.
        """
        self.eval() # prep model for evaluation
        server_loss = 0
        if (device!= None): self.to(device)
        for data, target in dataloader:
            if(device!= None):
                data = data.to(device)
                target = target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self(data)
            # calculate the loss
            loss = loss_func(output[0], target)
            # update running validation loss
            server_loss += loss.item()/data.size(0)
        server_loss = server_loss/len(dataloader)
        return server_loss

    def train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),
                    epochs = 1,device=None):
        """Trains the models a set number of epochs.

            Arguments:
                - dataloader: The training data in the form of a Pytorch Dataloader.
                - optimizer: The PyTorch implementation of a optimizer.
                - loss_func: The PyTorch implementation of a loss function (Default: Categorical Cross Entropy).
                - epochs: The number of epochs to run (Default: 1)
        """
        if (device!= None): self.to(device)
        self.train()
        for epoch in range(epochs):
            for i, (input_data, labels) in enumerate(dataloader):
                if(device!= None):
                    input_data = input_data.to(device)
                    labels = labels.to(device)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(input_data)   # batch x
                b_y = Variable(labels)   # batch y
                output = self(b_x)[0]
                loss = loss_func(output, b_y)
                # clear gradients for this training step
                optimizer.zero_grad()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
        return loss.item()

    def get_weights(self):
        return  copy.deepcopy(self.state_dict())

    def set_weights(self, model_state):
        self.load_state_dict(copy.deepcopy(model_state))
