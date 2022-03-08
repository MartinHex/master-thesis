from torch import nn
from torch.autograd import Variable
import copy

class MNIST_Model(nn.Module):
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

    def predict(self, input, device = None):
        self.eval()
        if (device!= None): self.to(device)
        if (device != None): input = input.to(device)
        output = self(input)
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return output

    def evaluate(self,dataloader,loss_func, device = None,take_mean=True):
        self.eval() # prep model for evaluation
        loss_per_batch = []
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
            loss_per_batch.append(loss.item()/data.size(0))
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        if(take_mean):
            return torch.mean(loss_per_batch)
        else:
            return loss_per_batch

    def train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),
                    epochs = 1,device=None, generator = False):
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
                if generator: yield self.get_weights()
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return loss.item()

    def get_weights(self):
        return  copy.deepcopy(self.state_dict())

    def set_weights(self, model_state):
        self.load_state_dict(copy.deepcopy(model_state))
