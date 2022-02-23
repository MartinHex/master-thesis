import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import defaultdict
import numpy as np

class Callbacks():
    def __init__(self, dataloader, verbose = True):
        self.dataloader = dataloader
        self.verbose = verbose

    def server_loss(self, algorithm):
        test_loss = algorithm.server.evaluate(self.dataloader)
        if self.verbose:
            print("Server val loss: {:.5f}".format(test_loss))
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
            print("Server val accuracy: {:.2f}".format(accuracy))
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

    def client_recall(self, algorithm):
        client_recalls = dict()
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            recall = _recall(client.model, self.dataloader)
            client_recalls['Client_{}'.format(i)] = recall
            if self.verbose:
                print("Client {} average val recall: {:.2f}".format(i, np.sum(recall) / (len(recall))))
        return client_recalls

    def server_recall(self, algorithm):
        algorithm.server.model.eval()
        recall = _recall(algorithm.server.model, self.dataloader)
        if self.verbose:
            print("Server average val recall: {:.2f}".format(np.sum(recall) / (len(recall))))
        return {'server': recall}

    def client_precision(self, algorithm):
        client_precision = dict()
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            precision = _precision(client.model, self.dataloader)
            client_precision['Client_{}'.format(i)] = precision
            if self.verbose:
                print("Client {} average val precision: {:.2f}".format(i, np.sum(precision) / (len(precision))))
        return client_precision

    def server_precision(self, algorithm):
        algorithm.server.model.eval()
        precision = _precision(algorithm.server.model, self.dataloader)
        if self.verbose:
            print("Server average val precision: {:.2f}".format(np.sum(precision) / (len(precision))))
        return {'server': precision}

    def client_f1(self, algorithm):
        client_f1 = dict()
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            f1 = _f1(client.model, self.dataloader)
            client_f1['Client_{}'.format(i)] = f1
            if self.verbose:
                print("Client {} average val f1 score: {:.2f}".format(i, np.sum(f1) / (len(f1))))
        return client_f1

    def server_f1(self, algorithm):
        algorithm.server.model.eval()
        f1 = _f1(algorithm.server.model, self.dataloader)
        if self.verbose:
            print("Server average val f1 score: {:.2f}".format(np.sum(f1) / (len(f1))))
        return {'server': f1}

def _accuracy(model, dataloader):
    acc = 0
    for data, target in dataloader:
        output = model(data)
        output_labels = torch.argmax(output[0], axis = -1)
        acc += torch.sum(output_labels == target)/data.size(0)
    return acc.item() / len(dataloader)

def _recall(model, dataloader):
    target_true = torch.Tensor()
    target_pred = torch.Tensor()
    for data, target in dataloader:
        output = model(data)
        output_labels = torch.argmax(output[0], axis = -1)
        target_true = torch.cat((target_true, target))
        target_pred = torch.cat((target_pred, output_labels))
    return list(recall_score(target_true, target_pred, average = None))

def _precision(model, dataloader):
    target_true = torch.Tensor()
    target_pred = torch.Tensor()
    for data, target in dataloader:
        output = model(data)
        output_labels = torch.argmax(output[0], axis = -1)
        target_true = torch.cat((target_true, target))
        target_pred = torch.cat((target_pred, output_labels))
    return list(precision_score(target_true, target_pred, average = None))

def _f1(model, dataloader):
    target_true = torch.Tensor()
    target_pred = torch.Tensor()
    for data, target in dataloader:
        output = model(data)
        output_labels = torch.argmax(output[0], axis = -1)
        target_true = torch.cat((target_true, target))
        target_pred = torch.cat((target_pred, output_labels))
    return list(f1_score(target_true, target_pred, average = None))
