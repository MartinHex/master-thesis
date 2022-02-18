from Servers.FedAgServer import FedAgServer
from Clients.FedAvgClient import FedAvgClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedAg(ABCAlgorithm):
    def __init__(self,dataloader,Model,callbacks=None,n_clients=5):
        super().__init__(n_clients,dataloader,Model, callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(n_clients)
        self.clients = [FedAvgClient(Model(), loader) for loader in client_dataloaders]
        self.server = FedAgServer(Model())
