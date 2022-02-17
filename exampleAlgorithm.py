from Models.StackOverflow_Model import StackOverflow_Model as Model
from Dataloaders.StackOverflow import StackOverflow as Dataloader
from Algorithms.FedAvg import FedAvg

number_of_clients = 5
batch_size = 16

dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)
def server_loss(self):
    test_loss = self.server.evaluate(test_data)
    print("Current Server val loss: {:.5f}".format(test_loss))

def client_loss(self):
    for i, client in enumerate(self.clients):
        client_loss = client.evaluate(test_data)
        print("Client {}: val loss: {:.5f}".format(i, client_loss))

callbacks = [
    client_loss,
    server_loss,
]

alg = FedAvg(dataloader=dataloader, Model=Model, callbacks = callbacks, n_clients=5)

alg.run(1)
