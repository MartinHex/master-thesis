from Servers.FedKpServer import FedKpServer
from Clients.FedPaClient import FedPaClient
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm

class FedKpPa(Algorithm):
    def __init__(self,dataloader,Model,
            batch_size=16,
            clients_per_round=None,
            store_distributions = False,
            client_lr = 0.01,
            client_burnin =  0,
            shrinkage = 1,
            momentum=0,
            decay=0,
            dampening=0,
            cluster_mean = True,
            kernel_function = 'epanachnikov',
            bandwidth = 'silverman',
            server_optimizer='none',
            server_lr=1,
            tau=0.1,
            b1=.9,
            b2=0.99,
            server_momentum=1,
            clients_sample_alpha = 'inf',
            max_iter=100,
            seed=1234,
            burnin=0
            ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)

        if(burnin<0 and not isinstance(burnin, int)):
            raise Exception('Invalid value of burnin.')
        self.burnin=burnin

        def client_generator(dataloader,round):
            if round<self.burnin:
                return SGDClient(Model(), dataloader,
                                    learning_rate=client_lr,
                                    momentum=momentum,
                                    decay=decay,
                                    dampening=dampening)
            else:
                return FedPaClient(Model(), dataloader,
                                    learning_rate = client_lr,
                                    burn_in =  client_burnin,
                                    shrinkage = shrinkage,
                                    momentum=momentum,
                                    decay=decay,
                                    dampening=dampening)
        server = FedKpServer(Model(),
                            store_distributions = store_distributions,
                            kernel_function = kernel_function,
                            cluster_mean = cluster_mean,
                            bandwidth=bandwidth,
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum,
                            max_iter=max_iter)

        super().__init__(server, client_dataloaders,client_generator=client_generator,clients_per_round=clients_per_round,seed=seed, clients_sample_alpha = clients_sample_alpha)
