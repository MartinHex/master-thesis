from Servers.FedKpServer import FedKpServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm

class FedKp(Algorithm):
    def __init__(self,dataloader,Model,callbacks=None, save_callbacks = False,batch_size=16,clients_per_round=None):
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        server = FedKpServer(Model())
        super().__init__(server,SGDClient, Model, client_dataloaders, callbacks, save_callbacks,clients_per_round=clients_per_round)
