from Servers.FedBeServer import FedBeServer
from Clients.FedAvgClient import FedAvgClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedBe(ABCAlgorithm):
    def __init__(self,dataloader,Model,loc_data,callbacks=None,n_clients=5, save_callbacks = False):
        super().__init__(n_clients,dataloader,Model, callbacks, save_callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(n_clients)
        self.clients = [FedAvgClient(Model(), loader) for loader in client_dataloaders]
        self.server = FedBeServer(Model(),loc_data)
