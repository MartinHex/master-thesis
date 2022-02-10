from Servers.FedAvgServer import FedAvgServer
from Clients.FedAvgClient import FedAvgClient
from Architectures.MNIST_Architecture import MNIST_Architecture
from Dataloaders.Mnist import Mnist
from Algorithms.FedAvg import FedAvg

mnist = Mnist(5)

test_data = mnist.get_training_dataloaders(5)[0]
def callbackExample(algorithm):
    test_loss = self.server.model.eval(test_data)
    print("Current Server val loss: %.5f"%test_loss)
    
alg = FedAvg(dataloader=mnist,Architecture=MNIST_Architecture,callback=callbackExample,n_clients=5)

alg.run(1)
