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

# Parameters
alpha=0.1
number_of_clients = 400
bandwidth_methods = ['silverman','local','scott','plugin','crossval']
batch_size = 16
hs = torch.logspace(-2,1,20)
cv = 10
out_path = os.path.join('data','Results','MNIST_bandwidth_evaluation_clustring')
log_path = os.path.join(out_path,'logs')
plot_path = os.path.join(out_path,'plots')
if not os.path.exists(out_path): os.mkdir(out_path)
if not os.path.exists(out_path): os.mkdir(log_path)
if not os.path.exists(out_path): os.mkdir(plot_path)

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


res = {}
for bandwidth_selection in bandwidth_methods:
    fedKp = FedKpServer(Model(),cluster_mean=False,bandwidth=bandwidth_selection)
    res = defaultdict(list)
    for i in range(cv):
        print('------------------- Round %i --------------------------'%i)
        print('setting up initial model')
        init_Model = Model()
        clients = [Client(Model(),dl) for dl in train_dls[:20]]
        # train Clients
        print('Training clients')
        for client in tqdm(clients):
            client.set_weights(init_Model.get_weights())
            client.train()

        print('Evaluating FedAvg')
        fedAvg.aggregate(clients)
        fedAvg_acc = server_accuracy(fedAvg,test_data,device)
        fedAvg_loss = server_loss(fedAvg,test_data,device)
        res['fedAvg_acc'].append(fedAvg_acc)
        res['fedAvg_loss'].append(fedAvg_loss)

        print('Testing FedKp')
        kp_losses = []
        kp_accuracies = []
        for h in tqdm(hs):
            fedKp.bandwidth_scaling=h
            fedKp.aggregate(clients)
            kp_accuracies.append(server_accuracy(fedKp,test_data,device))
            kp_losses.append(server_loss(fedKp,test_data,device))

        res['fedKp_losses'].append(kp_losses)
        res['fedKp_accuracies'].append(kp_accuracies)

    with open(os.path.join(log_path,bandwidth_selection+'.json'),'w') as f:
        json.dump(res,f)

############################# Plotting ##############################
#Load file
for json_file in  os.listdir(log_path):
    if json_file.split('.')[-1]!='json': continue
    with open(os.path.join(out_path,json_file),'r') as f:
        res = json.load(f)
    name = json_file.split('.')[0]
    # Plot losses
    fedAvg_m = np.array([np.mean(res['fedAvg_loss']) for _ in range(len(hs))])
    fedAvg_std = np.array([np.std(res['fedAvg_loss']) for _ in range(len(hs))])
    fedKp_m = np.mean(res['fedKp_losses'],0)
    fedKp_std = np.std(res['fedKp_losses'],0)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(hs,fedAvg_m,color='blue',label='FedAvg')
    ax.fill_between(hs,fedAvg_m-fedAvg_std,fedAvg_m+fedAvg_std,color='blue',alpha=0.2)
    ax.plot(hs,fedKp_m,color='red',label='FedKp')
    ax.fill_between(hs,fedKp_m-fedKp_std,fedKp_m+fedKp_std,color='red',alpha=0.2)
    ax.set_title('Server Loss')
    ax.set_xlabel('h')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(os.path.join(plot_path,name+'_server_loss.png'))
    plt.show()

    # Plot losses
    fedAvg_m = np.array([np.mean(res['fedAvg_acc']) for _ in range(len(hs))])
    fedAvg_std = np.array([np.std(res['fedAvg_acc']) for _ in range(len(hs))])
    fedKp_m = np.mean(res['fedKp_accuracies'],0)
    fedKp_std = np.std(res['fedKp_accuracies'],0)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(hs,fedAvg_m,color='blue',label='FedAvg')
    ax.fill_between(hs,fedAvg_m-fedAvg_std,fedAvg_m+fedAvg_std,color='blue',alpha=0.2)
    ax.plot(hs,fedKp_m,color='red',label='FedKp')
    ax.fill_between(hs,fedKp_m-fedKp_std,fedKp_m+fedKp_std,color='red',alpha=0.2)
    ax.set_title('Server Accuracies')
    ax.set_xlabel('h')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(os.path.join(plot_path,name+'_serveraccuracy.png'))
    plt.show()
