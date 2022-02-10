from Architectures.MNIST_Architecture import MNIST_Architecture
from Dataloaders.Mnist import Mnist
from Algorithms.FedAvg import FedAvg

number_of_clients = 5
batch_size = 16

mnist = Mnist(number_of_clients)

test_data = mnist.get_test_dataloader(batch_size)
def callback(self):
    test_loss = self.server.model.evaluate(test_data)
    print("Current Server val loss: %.5f"%test_loss)

alg = FedAvg(dataloader=mnist,Architecture=MNIST_Architecture,callback=callback,n_clients=5)

alg.run(1)
