from Servers.FedBeServer import FedBeServer
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm
from torch.utils.data import DataLoader

class FedBe(Algorithm):
    def __init__(self,dataloader,Model,callbacks=[], save_callbacks = False,p_validation=0.1,batch_size=16,clients_per_round=None):
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
        client = SGDClient(Model(), None)
        server = FedBeServer(Model(),loc_data)
        super().__init__(server, client, client_dataloaders_adj, callbacks, save_callbacks,clients_per_round=clients_per_round)
