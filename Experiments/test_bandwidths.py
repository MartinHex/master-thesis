import sys
sys.path.append('.')
from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Clients.SGDClient import SGDClient as Client
from Servers.FedKpServer import FedKpServer
from Servers.FedAvgServer import FedAvgServer
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import json
from random import sample

# Parameters
cluster_mean = False
loc_epochs = 5
alpha=0.01
number_of_clients = 100
clients_per_round = 20
bandwidth_methods = []#['silverman','local','plugin','crossval']
batch_size = 16
hs = torch.logspace(-2,1,20)
cv = 5
out_path = os.path.join('data','Results','MNIST','bandwidth_evaluation')
log_path = os.path.join(out_path,'logs')
plot_path = os.path.join(out_path,'plots')
if not os.path.exists(out_path): os.mkdir(out_path)
if not os.path.exists(log_path): os.mkdir(log_path)
if not os.path.exists(plot_path): os.mkdir(plot_path)

# Variables
dataloader = Dataloader(number_of_clients,alpha=alpha)
test_data = dataloader.get_test_dataloader(batch_size)
train_dls = dataloader.get_training_dataloaders(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create servers
fedAvg = FedAvgServer(Model())

# Evaluate algorithms
def server_accuracy(server, dataloader, device):
    acc = 0
    for data, target in dataloader:
        output = server.model.predict(data, device = device)
        output_labels = torch.argmax(output[0], axis = -1).to('cpu')
        acc += torch.sum(output_labels == target)
    return acc.item() / len(dataloader.dataset)

def server_loss(server,dataloader,device):
    test_loss = server.evaluate(dataloader, device = device,take_mean=False)
    test_loss = np.sum(test_loss)
    return test_loss

def train_loss(server,clients,device):
    res = [0]*len(clients)
    for i,client in enumerate(clients):
        res[i] = server_accuracy(server,client.dataloader,device)
    return res

def train_acc(server,clients,device):
    res = [0]*len(clients)
    for i,client in enumerate(clients):
        res[i] = server_loss(server,client.dataloader,device)
    return res


res = {}
for bandwidth_selection in bandwidth_methods:
    fedKp = FedKpServer(Model(),cluster_mean=cluster_mean,bandwidth=bandwidth_selection)
    res = defaultdict(list)
    for i in range(cv):
        print('------------------- Round %i --------------------------'%i)
        print('setting up initial model')
        init_Model = Model()
        clients = [Client(Model(),dl, learning_rate = 0.01,momentum=0.9) for dl in sample(train_dls,clients_per_round)]
        # train Clients
        print('Training clients')
        for client in tqdm(clients):
            client.set_weights(init_Model.get_weights())
            client.train(loc_epochs)

        print('Evaluating FedAvg')
        fedAvg.aggregate(clients)
        res['fedAvg_acc'].append(server_accuracy(fedAvg,test_data,device))
        res['fedAvg_loss'].append(server_loss(fedAvg,test_data,device))
        res['fedAvg_train_acc'].append(train_loss(fedAvg,clients,device))
        res['fedAvg_train_loss'].append(train_acc(fedAvg,clients,device))

        print('Testing FedKp')
        kp_losses = []
        kp_accuracies = []
        kp_train_loss = []
        kp_train_acc = []
        for h in tqdm(hs):
            fedKp.bandwidth_scaling=h
            fedKp.aggregate(clients)
            kp_accuracies.append(server_accuracy(fedKp,test_data,device))
            kp_losses.append(server_loss(fedKp,test_data,device))
            kp_train_acc.append(train_acc(fedKp,clients,device))
            kp_train_loss.append(train_loss(fedKp,clients,device))

        res['fedKp_loss'].append(kp_losses)
        res['fedKp_acc'].append(kp_accuracies)
        res['fedKp_train_loss'].append(kp_losses)
        res['fedKp_train_acc'].append(kp_accuracies)

    with open(os.path.join(log_path,bandwidth_selection+'.json'),'w') as f:
        json.dump(res,f)

############################# Plotting ##############################

def plot_func(key,name):
    fedAvg_m = np.array([np.mean(res['fedAvg_%s'%key]) for _ in range(len(hs))])
    fedAvg_std = np.array([np.std(res['fedAvg_%s'%key]) for _ in range(len(hs))])
    fedKp_m = np.mean(res['fedKp_%s'%key],0)
    fedKp_std = np.std(res['fedKp_%s'%key],0)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(hs,fedAvg_m,color='blue',label='FedAvg')
    ax.fill_between(hs,fedAvg_m-fedAvg_std,fedAvg_m+fedAvg_std,color='blue',alpha=0.2)
    ax.plot(hs,fedKp_m,color='red',label='FedKp')
    ax.fill_between(hs,fedKp_m-fedKp_std,fedKp_m+fedKp_std,color='red',alpha=0.2)
    ax.set_title(key)
    ax.set_xlabel('h')
    ax.set_ylabel(key)
    ax.set_xscale('log')
    ax.legend()
    out_fldr = os.path.join(plot_path,name)
    if not os.path.exists(out_fldr): os.mkdir(out_fldr)
    plt.savefig(os.path.join(out_fldr,'%s.png'%key))
    plt.show()

keys = ['loss','acc','train_loss','train_acc']

#Load file
for json_file in  os.listdir(log_path):
    with open(os.path.join(log_path,json_file),'r') as f:
        res = json.load(f)
    name = json_file.split('.')[0]
    # Plot losses
    for key in keys:
        plot_func(key,name)
