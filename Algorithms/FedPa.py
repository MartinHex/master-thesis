from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm
from datetime import datetime
import copy
from tqdm import tqdm
import os
from collections import defaultdict

class FedPa(Algorithm):
    def __init__(self,dataloader,Model,
                batch_size=16,
                clients_per_round=None,
                client_lr = 0.01,
                client_burnin =  0,
                shrinkage = 0.01,
                momentum=0,
                decay=0,
                dampening=0,
                server_optimizer='none',
                server_lr=1,
                tau=0.0,
                b1=.9,
                b2=0.99,
                server_momentum=0,
                burnin=0,
                clients_sample_alpha = 'inf',
                seed=1234
                ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)


        if(burnin<0 and not isinstance(burnin, int)):
            raise Exception('Invalid value of burnin.')
        self.burnin=burnin

        def client_generator(dataloader,round):
            if round<self.burnin:
                return SGDClient(Model(), dataloader,
                                    learning_rate=client_lr,
                                    momentum=momentum,
                                    decay=momentum,
                                    dampening=dampening)
            else:
                return FedPaClient(Model(), dataloader,
                                    learning_rate = client_lr,
                                    burn_in =  client_burnin,
                                    shrinkage = shrinkage,
                                    momentum=momentum,
                                    decay=decay,
                                    dampening=dampening)

        server = FedAvgServer(Model(),
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum)

        super().__init__(server, client_dataloaders,client_generator=client_generator,seed=seed,clients_per_round=clients_per_round, clients_sample_alpha = clients_sample_alpha)
