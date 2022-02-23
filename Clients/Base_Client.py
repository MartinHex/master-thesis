from abc import ABC, abstractmethod
from torch import nn
class Base_Client(ABC):
    def __init__(self, model):
        self.model = model

    def train(self,device=None):
        self.model.train_model(self.dataloader,self.optimizer,epochs = 1,device=device)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, model_state):
        self.model.set_weights(model_state)

    def evaluate(self, dataloader, loss_func = nn.CrossEntropyLoss()):
        return self.model.evaluate(dataloader, loss_func)
