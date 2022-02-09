from abc import ABC, abstractmethod
class Base_Client(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self):
        pass

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, model_state):
        self.model.set_weights(model_state)
