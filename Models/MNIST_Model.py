from torch import nn
from torch.autograd import Variable
import copy

class MNIST_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization

    def evaluate(self,dataloader,loss_func):
        self.eval() # prep model for evaluation
        server_loss = 0
        for data, target in dataloader:
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
        model = self.to(device) if (device!= None) else self
        model.train()
        for epoch in range(epochs):
            for i, (input_data, labels) in enumerate(dataloader):
                if(device!= None):
                    input_data = input_data.to(device)
                    labels = labels.to(device)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(input_data)   # batch x
                b_y = Variable(labels)   # batch y
                output = model(b_x)[0]
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
