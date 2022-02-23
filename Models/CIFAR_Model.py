from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet
import copy

class CIFAR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        number_of_classes = 100
        self.model = resnet.resnet18(pretrained = False, norm_layer=MyGroupNorm)
        self.model.fc = nn.Linear(512, number_of_classes)


    def forward(self, x):
        output = self.model(x)
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
        #def train(num_epochs, model, loader,optimizer,loss_func):
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

class MyGroupNorm(nn.Module):
    # epsilon = 0.001 seem to be default in FedPA implementation
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-3, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x
