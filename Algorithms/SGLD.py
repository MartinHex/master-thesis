from Servers.SGLDServer import SGLDServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm
import torch
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
from tqdm import tqdm
import json

class SGLD(Algorithm):
    def __init__(self,dataloader,Model,
                batch_size=16,
                clients_per_round=None,
                client_lr=0.1,
                momentum=0,
                decay=0,
                dampening=0,
                server_optimizer='none',
                server_lr=1,
                tau=0.1,
                b1=.9,
                b2=0.99,
                server_momentum=1,
                clients_sample_alpha = 'inf',
                burn_in = 0,
                ):
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)

        def client_generator(dataloader,round):
            return SGDClient(
                Model(),
                dataloader,
                learning_rate = client_lr,
                momentum = momentum,
                decay = decay,
                dampening = dampening
                )

        server = SGLDServer(
            Model(),
            Model(),
            burn_in,
            optimizer=server_optimizer,
            lr=server_lr,
            client_lr = client_lr,
            tau=tau,
            b1=b1,
            b2=b2,
            momentum = server_momentum
            )

        super().__init__(
            server,
            client_dataloaders,
            clients_per_round=clients_per_round,
            clients_sample_alpha = clients_sample_alpha,
            client_generator = client_generator,
            )
