from abc import ABC,abstractmethod
from collections import defaultdict

class ABCAlgorithm(ABC):

    def __init__(self,n_clients,dataloader,Model,callbacks):
        self.n_clients=n_clients
        self.dataloader=dataloader
        self.Model = Model
        self.callbacks = callbacks
        self.callback_data = [defaultdict(lambda: []) for i in range(len(self.callbacks))]

    def run(self,iterations):
        self.server.push_weights(self.clients)
        for round in range(iterations):
            print('---------------- Round {} ----------------'.format(round + 1))
            for client in self.clients:
                loss = client.train()
            self.server.aggregate(self.clients)
            self._run_callbacks() if (self.callbacks != None) else None
            self.server.push_weights(self.clients)
        return None

    def _run_callbacks(self):
        for index, callback in enumerate(self.callbacks):
            new_values = callback(self)
            for key, value in new_values.items():
                self.callback_data[index][key].append(value)
        return None
