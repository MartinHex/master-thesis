from Servers.FedBeServer import FedBeServer
from Clients.FedAvgClient import FedAvgClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedBe(ABCAlgorithm):
    def __init__(self,dataloader,Model,loc_data,callbacks=None,n_clients=5,M=10,
                    swa_lr1=0.001,swa_lr2=0.0004,swa_batch_size=20,
                    swa_freq=25,seed=1,swa_epochs=1,verbose=True):
        super().__init__(n_clients,dataloader,Model, callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(n_clients)
        self.clients = [FedAvgClient(Model(), loader) for loader in client_dataloaders]
        self.server = FedBeServer(Model(),loc_data,M,swa_lr1,swa_lr2,swa_batch_size,
                        swa_freq,seed,swa_epochs,verbose)
