from Servers.FedKpServer import FedKpServer
from Clients.FedPaClient import FedPaClient
from Algorithms.Algorithm import Algorithm

class FedKpPa(Algorithm):
    def __init__(self,dataloader,Model,
            batch_size=16,
            clients_per_round=None,
            store_distributions = False,
            client_lr = 0.01,
            client_burnin =  0,
            K = 1,
            shrinkage = 1,
            mcmc_samples = 1,
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

        if clients_per_round==None:
            clients = [FedPaClient(Model(), None,
                            learning_rate = client_lr,
                            burn_in =  client_burnin,
                            K = K,
                            shrinkage = shrinkage,
                            mcmc_samples = mcmc_samples,
                            momentum=momentum,
                            decay=momentum,
                            dampening=dampening)for _ in range(len(client_dataloaders))]
        else:
            clients = [FedPaClient(Model(), None,
                            learning_rate = client_lr,
                            burn_in =  client_burnin,
                            K = K,
                            shrinkage = shrinkage,
                            mcmc_samples = mcmc_samples,
                            momentum=momentum,
                            decay=momentum,
                            dampening=dampening) for _ in range(clients_per_round)]

        server = FedKpServer(Model(),
                            store_distributions = store_distributions,
                            cluster_mean = cluster_mean,
                            bandwidth=bandwidth,
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum,
                            max_iter=max_iter)

        super().__init__(server, client, client_dataloaders,clients_per_round=clients_per_round,seed=seed, clients_sample_alpha = clients_sample_alpha)
