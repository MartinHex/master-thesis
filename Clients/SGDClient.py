from Clients.Base_Client import Base_Client
from torch import optim

class SGDClient(Base_Client):
    def __init__(self, model, dataloader, learning_rate = 0.01,momentum=0,decay=0,dampening=0):
        super(SGDClient, self).__init__(model)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.dampening = dampening
        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate,
                                    momentum=momentum,weight_decay=decay,dampening=dampening)
        self.dataloader = dataloader

    def reset_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate,
            momentum = self.momentum, weight_decay = self.decay, dampening = self.dampening)
