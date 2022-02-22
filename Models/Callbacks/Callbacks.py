import torch
from collections import defaultdict

class Callbacks():
    def __init__(self, dataloader, verbose = True):
        self.dataloader = dataloader
        self.verbose = verbose

    def server_loss(self, algorithm):
        test_loss = algorithm.server.evaluate(self.dataloader)
        if self.verbose:
            print("Current Server val loss: {:.5f}".format(test_loss))
        return {'server': test_loss}

    def client_loss(self, algorithm):
        client_losses = dict()
        for i, client in enumerate(algorithm.clients):
            client_loss = client.evaluate(self.dataloader)
            client_losses['Client_{}'.format(i)] = client_loss
            if self.verbose:
                print("Client {}: val loss: {:.5f}".format(i, client_loss))
        return client_losses

    def server_accuracy(self, algorithm):
        algorithm.server.model.eval()
        accuracy = _accuracy(algorithm.server.model, self.dataloader)
        if self.verbose:
            print("Current Server val accuracy: {:.2f}".format(accuracy))
        return {'server': accuracy}

    def client_accuracy(self, algorithm):
        client_accuracies = dict()
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            accuracy = _accuracy(client.model, self.dataloader)
            client_accuracies['Client_{}'.format(i)] = accuracy
            if self.verbose:
                print("Client {} val accuracy: {:.2f}".format(i, accuracy))
        return client_accuracies

def _accuracy(model, dataloader):
    for data, target in dataloader:
        output = model(data)
        output_labels = torch.argmax(output[0], axis = -1)
        acc = torch.sum(output_labels == target)/data.size(0)
    return acc / len(dataloader)
