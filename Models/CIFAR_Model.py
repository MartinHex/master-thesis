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

    def predict(self, input, device = None):
        self.eval()
        if (device!= None): self.to(device)
        if (device != None): input = input.to(device)
        self(input)
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return output

    def evaluate(self,dataloader,loss_func, device = None, take_mean=True):
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
                    epochs = 1,device=None):
        #def train(num_epochs, model, loader,optimizer,loss_func):
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
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
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
