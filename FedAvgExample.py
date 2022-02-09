from Servers.FedAvgServer import FedAvgServer
from Clients.FedAvgClient import FedAvgClient
from Architectures.MNIST_Architecture import MNIST_Architecture
from Dataloaders.Mnist import Mnist

number_of_clients = 5
number_of_rounds = 1
batch_size = 16

mnist = Mnist(number_of_clients)
client_dataloaders = mnist.get_training_dataloaders(batch_size)
clients = [FedAvgClient(MNIST_Architecture(), loader) for loader in client_dataloaders]
server = FedAvgServer(MNIST_Architecture())

for round in range(number_of_rounds):
    for client in clients:
        loss = client.train()
        print(loss)

    server.aggregate(clients)
    server.push_weights(clients)
