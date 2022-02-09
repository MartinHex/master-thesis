from pathlib import Path
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import math
from torchvision import models,datasets
import copy
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import optim

# Load dataset
mnist=datasets.MNIST('./MNIST',
    transform = ToTensor(), download=True)

# Select subset to speed up computation
mnist = data_utils.Subset(mnist,torch.arange(5000))

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
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

# Define larger datasets
loss_func = nn.CrossEntropyLoss()
loss_func

#Optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.01)




######################### Federated learning ##################################

# Parameters
n_clients = 5
loc_epochs = 1
step_length=0.01
iters = 10

# Set variables
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = step_length)

# Set seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set initial models
centralized_model = CNN_model()
server_model = CNN_model()
client_Models = [CNN_model() for i in range(n_clients)]

#Set dataloaders
centralized_loader =DataLoader(mnist,batch_size=100,num_workers=1)
# Client loaders
n=len(mnist)
splits = [len(mnist)//n_clients for _ in range(n_clients)]
data_splits = torch.utils.data.random_split(mnist,splits)
client_loaders = [DataLoader(data,batch_size=100,num_workers=1)
                    for data in data_splits]

# Training function
def train(num_epochs, model, loader,optimizer,loss_func):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
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


res = {'centralized_loss':[],'server_loss':[],'clientloss':[[] for _ in range(n_clients)]}
for iter in range(iters):
    # Train centralized model
    centralized_loss = train(loc_epochs,centralized_model,centralized_loader,
                    optimizer,loss_func)

    # Train Clients
    client_loss = [0 for i in range(n_clients)]
    for i in range(n_clients):
        client_loss[i]=train(loc_epochs,client_Models[i],client_loaders[i],
                        optimizer,loss_func)
    client_loss
    # Aggregate through fedAvg
    server_state = server_model.state_dict()
    client_states = [cl.state_dict() for cl in client_Models]
    for key in server_state:
        client_tens = [state[key] for state in client_states]
        server_state[key] = torch.stack(client_tens, dim=0).sum(dim=0)/n_clients

    # Update state
    server_model.load_state_dict(server_state)
    # Broadcast State
    for i in range(n_clients):
        client_Models[i].load_state_dict(copy.deepcopy(server_model.state_dict()))

    server_model.eval() # prep model for evaluation
    server_loss = 0
    for data, target in centralized_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = server_model(data)
        # calculate the loss
        loss = loss_func(output[0], target)
        # update running validation loss
        server_loss += loss.item()/data.size(0)
    server_loss = server_loss/len(centralized_loader)

    # Store results
    res['centralized_loss']+=[centralized_loss]
    res['server_loss']+=[server_loss]
    for i in range(n_clients):
        res['clientloss'][i] += [client_loss[i]]

    print("Iteration %i: Centralized loss: %.5f, server_loss:%.5f, client loss: %s."%(
        iter,centralized_loss,server_loss,str(client_loss)))
