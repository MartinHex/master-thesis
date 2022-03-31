from abc import ABC,abstractmethod
from torch import nn
import torch
import copy
from torch.autograd import Variable

class nn_Model(nn.Module):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


    def predict(self, input, device = None):
        self.eval()
        if (device!= None): self.to(device)
        if (device != None): input = input.to(device)
        output = self(input)
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return output

    def evaluate(self,dataloader,loss_func, device = None, take_mean=True):
        """Evaluates the model on a given loss function.

            Arguments:
                - dataloader: Dataloader with data to evaluate on.
                - loss_func: Loss function to use for evaluation.
        """
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
            return sum(loss_per_batch)/len(loss_per_batch)
        else:
            return loss_per_batch

    def train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),
                    epochs = 1,device=None):
        """Trains the models a set number of epochs.

            Arguments:
                - dataloader: The training data in the form of a Pytorch Dataloader.
                - optimizer: The PyTorch implementation of a optimizer.
                - loss_func: The PyTorch implementation of a loss function (Default: Categorical Cross Entropy).
                - epochs: The number of epochs to run (Default: 1)
        """
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


    def iter_train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),
                    epochs = 1,device=None):
        """Trains the models a set number of epochs.

            Arguments:
                - dataloader: The training data in the form of a Pytorch Dataloader.
                - optimizer: The PyTorch implementation of a optimizer.
                - loss_func: The PyTorch implementation of a loss function (Default: Categorical Cross Entropy).
                - epochs: The number of epochs to run (Default: 1)
        """
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
                yield self.get_weights()
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return loss.item()

    def get_weights(self):
        return  copy.deepcopy(self.state_dict())

    def set_weights(self, model_state):
        self.load_state_dict(copy.deepcopy(model_state))

    def reset_model(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
