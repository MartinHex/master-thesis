from abc import ABC,abstractmethod

class ABCServer(ABC):

    def __init__(self,model):
        self.model = model

    @abstractmethod
    def aggregate(self, clients):
        pass

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def push_weights(self, clients):
        for client in clients:
            client.set_weights(self.model.get_weights())

    def evaluate(self, dataloader):
        return self.model.evaluate(dataloader)
