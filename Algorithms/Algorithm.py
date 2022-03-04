from abc import ABC,abstractmethod
from collections import defaultdict
import json
import os
from datetime import datetime
from Servers.ProbabilisticServer import ProbabilisticServer
from random import sample

class Algorithm():
    def __init__(self,server,clients, callbacks=None, save_callbacks=False,clients_per_round=None):
        self.callbacks = callbacks if callbacks!=None else []
        self.clients = clients
        self.clients_per_round = len(clients) if clients_per_round==None else clients_per_round
        if( len(clients)<self.clients_per_round):
            raise Exception('More clients per round than clients provided.')
        self.server = server
        self.save_callbacks = save_callbacks
        self.callback_data = dict()
        for name, callback in self.callbacks:
            self.callback_data[name] = defaultdict(lambda: [])
        self.callback_data['timestamps'] = []

    def run(self,iterations, epochs = 1, device = None,option = 'mle'):
        if(option not in ['mle','single_sample','multi_sample']):
            raise Exception("""Incorrect option provided must be either 'mle', 'single_sample' or 'multi_sample'""")
        self.start_time = datetime.now()
        self.server.push_weights(self.clients)
        for round in range(iterations):
            print('---------------- Round {} ----------------'.format(round + 1))
            self.callback_data['timestamps'].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            client_sample = self.sample_clients()
            for client in client_sample:
                loss = client.train(epochs = epochs, device = device)
            self.server.aggregate(client_sample)
            if (self.callbacks != None): self._run_callbacks()
            if option == 'mle' or not isinstance(self.server,ProbabilisticServer):
                self.server.push_weights(self.clients)
            elif option =='single_sample':
                self.server.set_weights(elf.server.sample_model())
                self.server.push_weights(self.clients)
            elif option =='multi_sample':
                for client in self.clients:
                    client.set_weights(self.server.sample_model())

        if self.save_callbacks: self._save_callbacks()
        return None

    def get_callback_data(self,key):
        return self.callback_data[key]

    def _run_callbacks(self):
        for name, callback in self.callbacks:
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

    def sample_clients(self):
        if self.clients_per_round!=len(self.clients):
            return sample(self.clients,self.clients_per_round)
        else:
            return self.clients
