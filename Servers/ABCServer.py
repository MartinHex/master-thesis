from abc import ABC,abstractmethod
from Servers.optimizers.adam import Adam
from Servers.optimizers.sgd import SGD
from torch import nn

class ABCServer(ABC):

    def __init__(self,model,optimizer='none',lr=1,tau=0.1,b1=.9,b2=0.99,momentum=1):
        self.model = model
        if optimizer == 'sgd':
            self.optimizer= SGD(lr=lr,momentum=momentum)
        elif optimizer == 'adam':
            self.optimizer= Adam(lr=lr,tau=tau,b1=b1,b2=b2)
        else:
            self.optimizer= SGD(lr = 1, momentum = 0)

    @abstractmethod
    def combine(self,clients):
        pass

    def aggregate(self, clients,device=None, client_scaling = None):
        w_old = self.get_weights()
        w_new = self.combine(clients,device=device, client_scaling = client_scaling)
        w_new = self.fedOpt(w_new,w_old)
        self.set_weights(w_new)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def push_weights(self, clients):
        for client in clients:
            client.set_weights(self.get_weights())

    def evaluate(self, dataloader, loss_func=nn.CrossEntropyLoss(), device = None, take_mean = True):
        return self.model.evaluate(dataloader, loss_func, device = device, take_mean = take_mean)

    def fedOpt(self,w_new,w_old):
        return self.optimizer.opt(w_new,w_old)
