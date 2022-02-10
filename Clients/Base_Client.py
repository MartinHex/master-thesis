from abc import ABC, abstractmethod
class Base_Client(ABC):
    def __init__(self, model):
        self.model = model

    def train(self):
        self.model.train_model(self.dataloader,self.optimizer,epochs = 1)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, model_state):
        self.model.set_weights(model_state)

    def evaluate(self, dataloader):
        return self.model.evaluate(dataloader)
