from Servers.FedAgServer import FedAgServer
from Clients.SGDClient import SGDClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedAg(ABCAlgorithm):
    def __init__(self,dataloader,Model,callbacks=None, save_callbacks = False,batch_size=16):
        super().__init__(dataloader,Model, callbacks, save_callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        self.clients = [SGDClient(Model(), loader) for loader in client_dataloaders]
        self.server = FedAgServer(Model())
