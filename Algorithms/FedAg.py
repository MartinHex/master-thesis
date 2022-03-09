from Servers.FedAgServer import FedAgServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm

class FedAg(Algorithm):
    def __init__(self,dataloader,Model,callbacks=None, save_callbacks = False,batch_size=16,clients_per_round=None):
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        client = SGDClient(Model(), None)
        server = FedAgServer(Model())
        super().__init__(server,client, client_dataloaders, callbacks, save_callbacks,clients_per_round=clients_per_round)
