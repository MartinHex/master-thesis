from Clients.Base_Client import Base_Client
from torch import optim

class SGDClient(Base_Client):
    def __init__(self, model, dataloader, learning_rate = 0.01):
        super(FedAvgClient, self).__init__(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        self.dataloader = dataloader
