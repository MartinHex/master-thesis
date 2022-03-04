from Servers.FedBeServer import FedBeServer
from Clients.SGDClient import SGDClient
from Algorithms.ABCAlgorithm import ABCAlgorithm
from torch.utils.data import DataLoader

class FedBe(ABCAlgorithm):
    def __init__(self,dataloader,Model,callbacks=[], save_callbacks = False,p_validation=0.1,batch_size=16):
        super().__init__(callbacks, save_callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(batch_size)
        loc_data = dataloader.get_test_dataloader(batch_size)
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
        # loc_data = DataLoader(loc)
        self.clients = [SGDClient(Model(), dl) for dl in client_dataloaders]
        self.server = FedBeServer(Model(),loc_data)
