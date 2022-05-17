from Servers.FedAvgServer import FedAvgServer
from Clients.FedProxClient import FedProxClient
from Algorithms.Algorithm import Algorithm

class FedProx(Algorithm):
    def __init__(self,dataloader,Model,
                batch_size=16,
                clients_per_round=None,
                client_lr=0.1,
                momentum=0,
                decay=0,
                dampening=0,
                server_optimizer='none',
                server_lr=1,
                tau=0.1,
                b1=.9,
                b2=0.99,
                server_momentum=1,
                clients_sample_alpha = 'inf',
                seed=1234,
                mu = 0,
                ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)

        def client_generator(dataloader,round):
            return FedProxClient(Model(), dataloader,
                                learning_rate=client_lr,
                                momentum=momentum,
                                decay=decay,
                                dampening=dampening,
                                mu = mu
                                )

        server = FedAvgServer(Model(),
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum)

        super().__init__(server, client_dataloaders,client_generator=client_generator,
                        clients_per_round=clients_per_round,seed=seed,
                         clients_sample_alpha = clients_sample_alpha)

    def _create_client(self,dataloader,round=0,i=0):
        return self.init_client(dataloader)
