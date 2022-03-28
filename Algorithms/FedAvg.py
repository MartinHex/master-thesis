from Servers.FedAvgServer import FedAvgServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm

class FedAvg(Algorithm):
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
                seed=1234
                ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)

        client = SGDClient(Model(), None,
                            learning_rate=client_lr,
                            momentum=momentum,
                            decay=momentum,
                            dampening=dampening)

        server = FedAvgServer(Model(),
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum)

        super().__init__(server, client, client_dataloaders, clients_per_round=clients_per_round,seed=seed, clients_sample_alpha = clients_sample_alpha)
