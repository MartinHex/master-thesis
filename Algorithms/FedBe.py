from Servers.FedBeServer import FedBeServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm
from torch.utils.data import DataLoader

class FedBe(Algorithm):
    def __init__(self,dataloader,Model,
                callbacks=None,
                save_callbacks = False,
                p_validation=0.1,
                batch_size=16,
                clients_per_round=None,
                client_lr=0.01,
                momentum=0,
                decay=0,
                dampening=0,
                M=10,
                swa_lr1=0.001,
                swa_lr2=0.0004,
                swa_freq=5,
                swa_epochs=1,
                server_optimizer='none',
                server_lr=1,
                tau=0.1,
                b1=.9,
                b2=0.99,
                server_momentum=1,
                clients_sample_alpha = 'inf',
                ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        # Set up local dataloader
        loc_data = []
        client_dataloaders_adj = []
        for loader in client_dataloaders:
            adj_dataloader = []
            loader_size = len(loader)
            for i,(x,y) in enumerate(loader):
                if(i<p_validation*loader_size):
                    loc_data.append((x,y))
                else:
                    adj_dataloader.append((x,y))
            client_dataloaders_adj.append(adj_dataloader)

        client = SGDClient(Model(), None,
                            learning_rate = client_lr,
                            momentum=momentum,
                            decay=momentum,
                            dampening=dampening)

        server = FedBeServer(Model(),loc_data,
                            M=M,
                            swa_lr1=swa_lr1,
                            swa_lr2=swa_lr2,
                            swa_freq=swa_freq,
                            swa_epochs=swa_epochs,
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum)

        super().__init__(server, client, client_dataloaders_adj, callbacks,
                            save_callbacks,clients_per_round=clients_per_round, clients_sample_alpha = clients_sample_alpha)
