from abc import ABC,abstractmethod
from Servers.ABCServer import ABCServer

class ProbabilisticServer(ABCServer):

    @abstractmethod
    def sample_model(self):
        pass
