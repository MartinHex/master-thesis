from Servers.FedKpServer import FedKpServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm

class FedKp(Algorithm):
    def __init__(self,dataloader,Model,
            batch_size=16,
            clients_per_round=None,
            store_distributions = False,
            bandwidth_scaling = 1,
            client_lr = 0.01,
            momentum=0,
            decay=0,
            dampening=0,
            cluster_mean = True,
            bandwidth = 'silverman',
            server_optimizer='none',
            server_lr=1,
            tau=0.1,
            b1=.9,
            b2=0.99,
            server_momentum=1,
            clients_sample_alpha = 'inf',
            max_iter=100,
            seed=1234
            ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)

        def client_generator(dataloader,round):
            return SGDClient(Model(), dataloader,
                                learning_rate=client_lr,
                                momentum=momentum,
                                decay=momentum,
                                dampening=dampening)

        server = FedKpServer(Model(),
                            store_distributions = store_distributions,
                            cluster_mean = cluster_mean,
                            bandwidth=bandwidth,
                            bandwidth_scaling=bandwidth_scaling,
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum,
                            max_iter=max_iter)

        super().__init__(server, client_dataloaders,client_generator=client_generator,clients_per_round=clients_per_round,seed=seed, clients_sample_alpha = clients_sample_alpha)
