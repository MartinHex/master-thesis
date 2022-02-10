from abc import ABC,abstractmethod

class ABCAlgorithm(ABC):

    def __init__(self,n_clients,dataloader,Architecture,callbacks):
        self.n_clients=n_clients
        self.dataloader=dataloader
        self.Architecture = Architecture
        self.callbacks = callbacks

    def run(self,iterations):
        for round in range(iterations):
            for client in self.clients:
                loss = client.train()
        self.server.aggregate(self.clients)
        if(self.callbacks != None):
            for callback in self.callbacks:
                callback(self)
        self.server.push_weights(self.clients)
