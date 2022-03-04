from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedPa(ABCAlgorithm):
    def __init__(self,dataloader,Model,callbacks=None, save_callbacks = False,batch_size=16):
        super().__init__(callbacks, save_callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        self.clients = [FedPaClient(Model(), loader) for loader in client_dataloaders]
        self.server = FedAvgServer(Model())
