from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Algorithms.FedAvg import FedAvg as Alg
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
callbacks = [ cbs.client_loss,cbs.server_loss,cbs.client_accuracy,cbs.server_accuracy]

#Create an instance of FedAvg and train a number of rounds
alg = Alg(dataloader=dataloader, Model=Model, batch_size=16)
alg.run(2, device = device,callbacks=callbacks,log_callbacks=False)

#Access the callback history and plot the client loss
server_loss = alg.get_callback_data()['server_loss']
plt.plot(server_loss)
plt.title('Server Loss')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.show()
