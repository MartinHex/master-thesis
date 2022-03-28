from Servers.FedBeServer import FedBeServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm
from torch.utils.data import DataLoader

class FedBe(Algorithm):
    def __init__(self,dataloader,Model,
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
                server_momentum=0,
                clients_sample_alpha = 'inf',
                seed=1234,
                ):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size=batch_size)
        # Set up local dataloader
        loc_data = []
        client_dataloaders_adj = []
        for client_dataloader in client_dataloaders:
            adj_dataloader = []
            test_size = len(client_dataloader.dataset)*p_validation
            count = 0
            for (x,y) in client_dataloader:
                for (xi,yi) in zip(x,y):
                    if(count<test_size):
                        loc_data.append((xi,yi))
                        count+=1
                    else:
                        adj_dataloader.append((xi,yi))
            adj_dataloader = DataLoader(adj_dataloader,batch_size=batch_size)
            client_dataloaders_adj.append(adj_dataloader)

        loc_data = DataLoader(loc_data,batch_size=batch_size,shuffle=True)

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

        super().__init__(server, client, client_dataloaders_adj,clients_per_round=clients_per_round,seed=seed, clients_sample_alpha = clients_sample_alpha)
