from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Algorithms.Algorithm import Algorithm

class FedPa(Algorithm):
    def __init__(self,dataloader,Model,callbacks=None, save_callbacks = False,
                batch_size=16,clients_per_round=None,client_lr = 0.01, burn_in =  0,
                K = 1, shrinkage = 1, mcmc_samples = 1,momentum=0,decay=0,dampening=0):
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        client = FedPaClient(Model(), None,learning_rate = client_lr, burn_in =  burn_in,
                        K = K, shrinkage = shrinkage, mcmc_samples = mcmc_samples,
                        momentum=momentum,decay=momentum,dampening=dampening)
        server = FedAvgServer(Model())
        super().__init__(server, client, client_dataloaders, callbacks, save_callbacks,clients_per_round=clients_per_round)
