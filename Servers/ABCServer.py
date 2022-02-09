from abc import ABC,abstractmethod

class ABCServer(ABC):

    def __init__(self,architecture):
        super.init()
        self.model = architecture

    @abstractmethod
    def aggregate(clients):
        pass

    def get_weights():
        return self.architecture.get_weights()

    def set_weights(weights):
        self.architecture.set_weights(weights)
