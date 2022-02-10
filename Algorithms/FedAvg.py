from Servers.FedAvgServer import FedAvgServer
from Clients.FedAvgClient import FedAvgClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedAvg(ABCAlgorithm):
    def __init__(self,dataloader,Architecture,callbacks=None,n_clients=5):
        super().__init__(n_clients,dataloader,Architecture, callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(n_clients)
        self.clients = [FedAvgClient(Architecture(), loader) for loader in client_dataloaders]
        self.server = FedAvgServer(Architecture())
