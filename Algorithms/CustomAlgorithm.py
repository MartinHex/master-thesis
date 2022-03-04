from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class CustomAlgorithm(ABCAlgorithm):
    """
        A custom algorithm for modularity of input parameters.
        From a provided server and client the algorithms provides the same
        functionalities provided in other algorithms, but now more flexible
        with regards to input.

    """
    def __init__(self,server,clients,callbacks=None, save_callbacks = False,batch_size=16):
        super().__init__(callbacks, save_callbacks)
        self.clients = clients
        self.server = server
