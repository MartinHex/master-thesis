from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Algorithms.Algorithm import Algorithm

class FedPa(Algorithm):
    def __init__(self,dataloader,Model,callbacks=None, save_callbacks = False,batch_size=16,clients_per_round=None):
        super().__init__(callbacks, save_callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        clients = [FedPaClient(Model(), loader) for loader in client_dataloaders]
        server = FedAvgServer(Model())
        super().__init__(server,clients, callbacks, save_callbacks,clients_per_round=clients_per_round)
