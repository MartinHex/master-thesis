from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Algorithms.FedAvg import FedAvg
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt

number_of_clients = 5
batch_size = 16
dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)

# Create callback functions that are run at the end of every round
cbs = Callbacks(test_data)
callbacks = [
    cbs.client_loss,
    cbs.server_loss,
    cbs.client_accuracy,
    cbs.server_accuracy,
]

#Create an instance of FedAvg and train a number of rounds
alg = FedAvg(dataloader=dataloader, Model=Model, callbacks = callbacks, n_clients=number_of_clients)
alg.run(2)

#Access the callback history and plot the client loss
for key, values in alg.callback_data[0].items():
    plt.plot(values, label = key)
plt.title('Clinet Loss')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.show()
