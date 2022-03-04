from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Algorithms.FedAvg import FedAvg
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt
import torch

number_of_clients = 5
batch_size = 16
dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
cbs = Callbacks(test_data, device = device)
callbacks = [
    ('client_loss', cbs.client_loss),
    ('server_loss', cbs.server_loss),
    ('client_accuracy', cbs.client_accuracy),
    ('server_accuracy', cbs.server_accuracy),
]

#Create an instance of FedAvg and train a number of rounds
alg = FedAvg(dataloader=dataloader, Model=Model, callbacks = callbacks, batch_size=16 save_callbacks = True)
alg.run(2, device = device)

#Access the callback history and plot the client loss
client_losses = alg.get_callback_data('client_loss')
for key, values in client_losses.items():
    plt.plot(values, label = key)
plt.title('Clinet Loss')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.show()
