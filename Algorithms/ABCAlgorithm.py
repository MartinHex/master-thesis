from abc import ABC,abstractmethod

class ABCAlgorithm(ABC):

    def __init__(self,n_clients,dataloader,Architecture,callback=None):
        self.n_clients=n_clients
        self.dataloader=dataloader
        self.Architecture = Architecture
        self.callback = callback

    def run(self,iterations):
        for round in range(iterations):
            for client in self.clients:
                loss = client.train()
        if(self.callback!=None):
            self.callback(self)
        self.server.aggregate(clients)
        self.server.push_weights(clients)
