from abc import ABC,abstractmethod
from collections import defaultdict
import json
import os
from datetime import datetime
from Servers.ProbabilisticServer import ProbabilisticServer
from random import sample
from tqdm import tqdm
import copy
import threading
import numpy as np

class Algorithm():
    def __init__(self,server,client, dataloaders, callbacks=None, save_callbacks=False,clients_per_round=None, clients_sample_alpha = 'inf'):
        self.callbacks = callbacks if callbacks!=None else []
        self.dataloaders = dataloaders
        self.clients_per_round = len(dataloaders) if clients_per_round==None else clients_per_round
        if( len(dataloaders)<self.clients_per_round):
            raise Exception('More clients per round than clients provided.')
        self.server = server
        self.save_callbacks = save_callbacks
        self.callback_data = dict()
        for name, callback in self.callbacks:
            self.callback_data[name] = defaultdict(lambda: [])
        self.callback_data['timestamps'] = []
        self.clients = [copy.deepcopy(client) for i in range(self.clients_per_round)]
        if clients_sample_alpha == 'inf':
            self.client_probabilities = [1/len(dataloaders)] * (len(dataloaders))
        else:
            assert (clients_sample_alpha >= 0 and clients_sample_alpha <= 1), 'Client sample alpha must be in (0,1).'
            self.client_probabilities = np.random.dirichlet([clients_sample_alpha] * len(dataloaders))

    def run(self,iterations, epochs = 1, device = None,option = 'mle',n_workers=3):
        if(option not in ['mle','single_sample','multi_sample']):
            raise Exception("""Incorrect option provided must be either 'mle', 'single_sample' or 'multi_sample'""")
        self.start_time = datetime.now()
        #self.server.push_weights(self.clients)
        for round in range(iterations):
            print('---------------- Round {} ----------------'.format(round + 1))
            self.callback_data['timestamps'].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            if option == 'single_sample': self.server.set_weights(self.server.sample_model())
            dataloader_sample = self.sample_dataloaders()

            for i, dataloader in enumerate(tqdm(dataloader_sample)):
                # Initialize Client to run
                self.clients[i].dataloader = dataloader
                self._set_model_weight(i, option)
                self.clients[i].reset_optimizer() # Reset optimizer so client momentum can be used, not momentum does not carry over between rounds
                loss = self.clients[i].train(epochs = epochs, device = device)

            dataloader_sizes = [len(dataloader.dataset) for dataloader in dataloader_sample]
            self.server.aggregate(self.clients, device=device, client_scaling = dataloader_sizes)
            if (self.callbacks != None): self._run_callbacks()

        if self.save_callbacks: self._save_callbacks()
        return None

    def _train_client(self,i,dataloader,option,epochs,device,losses):
        # Initialize Client to run
        self.clients[i].dataloader = dataloader
        self._set_model_weight(i, option)
        loss = self.clients[i].train(epochs = epochs, device = device)
        losses.append(loss)

    def get_callback_data(self,key):
        return self.callback_data[key]

    def _run_callbacks(self):
        for name, callback in self.callbacks:
            print('Running callback: {}'.format(name))
            new_values = callback(self)
            for key, value in new_values.items():
                self.callback_data[name][key].append(value)
        return None

    def _save_callbacks(self):
        file_name = 'experiment_{}.json'.format(self.start_time.strftime("%d_%m_%Y_%H_%M"))
        log_dir = os.path.join(os.getcwd(), 'data', 'logs')
        file_path = os.path.join(log_dir, file_name)
        if not os.path.exists(log_dir): os.mkdir(log_dir)
        with open(file_path, "w") as outfile:
            json.dump(self.callback_data, outfile)

    def _set_model_weight(self, i, option):
        if option == 'mle' or not isinstance(self.server,ProbabilisticServer):
            self.clients[i].set_weights(self.server.get_weights())
        elif option =='single_sample':
            self.clients[i].set_weights(self.server.get_weights())
        elif option =='multi_sample':
            self.clients[i].set_weights(self.server.sample_model())

    def sample_dataloaders(self):
        if self.clients_per_round!=len(self.dataloaders):
            dataloaders = np.random.choice(
                self.dataloaders,
                size = self.clients_per_round,
                replace = False,
                p = self.client_probabilities,
                )
            return dataloaders
        else:
            return self.dataloaders
