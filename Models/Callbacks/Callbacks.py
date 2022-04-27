import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from collections import defaultdict
from scipy.stats import kurtosis, skew, kstest
import numpy as np

class Callbacks():
    def __init__(self, dataloader, verbose = True, device = None):
        self.dataloader = dataloader
        self.verbose = verbose
        self.device = device

    def server_thesis_results(self, algorithm):
        (accuracy, recall, precision, loss) = _accrecprec(algorithm.server.model, self.dataloader, self.device)
        if self.verbose:
            print("Server val accuracy: {:.2f}".format(accuracy))
            print("Server val recall: {:.2f}".format(recall))
            print("Server val precision: {:.2f}".format(precision))
            print("Server val loss: {:.2f}".format(loss))
        return {'server_accuracy': accuracy, 'server_recall': recall, 'server_precision': precision, 'server_loss': loss}


    def ks_test(self, algorithm):
        client_weights = [_model_weight_to_array(c.get_weights()) for c in algorithm.clients]
        stacked_weights = torch.stack(client_weights, 0)
        means = torch.mean(stacked_weights, 0)
        std = torch.std(stacked_weights, 0)+0.00001
        ps = [kstest(stacked_weights[:,i], 'norm', (means[i], std[i]))[1] for i in range(len(means))]
        return {'ks_test':ps}

    def skew(self,algorithm):
        client_weights = [cl.get_weights() for cl in algorithm.clients]
        skews = torch.tensor(())
        for key in client_weights[0]:
            client_tensors = [state[key] for state in client_weights]
            stacked_weights = torch.stack(client_tensors, dim=0)
            indices = torch.arange(stacked_weights.shape[0])
            skews = torch.cat((skews, torch.flatten(torch.tensor(skew(torch.index_select(stacked_weights, 0, indices), axis = 0)))))
        return {'skew': [skew.item() for skew in skews]}

    def kurtosis(self,algorithm):
        client_weights = [cl.get_weights() for cl in algorithm.clients]
        kurtosises = torch.tensor(())
        for key in client_weights[0]:
            client_tensors = [state[key] for state in client_weights]
            stacked_weights = torch.stack(client_tensors, dim=0)
            indices = torch.arange(stacked_weights.shape[0])
            kurtosises = torch.cat((kurtosises, torch.flatten(torch.tensor(kurtosis(torch.index_select(stacked_weights, 0, indices), axis = 1)))))
        return {'kurtosis': [kurtosis.item() for kurtosis in kurtosises]}

    def server_loss(self, algorithm):
        test_loss = algorithm.server.evaluate(self.dataloader, device = self.device,take_mean=False)
        test_loss = np.sum(test_loss)
        if self.verbose:
            print("Server val loss: {:.5f}".format(test_loss))
        return {'server_loss': test_loss}

    def client_loss(self, algorithm):
        client_losses = []
        for i, client in enumerate(algorithm.clients):
            client_loss = client.evaluate(self.dataloader, device = self.device,take_mean=False)
            client_loss = np.sum(client_loss)
            client_losses.append(client_loss)
            if self.verbose:
                print("Sampled Client {} loss: {:.3f}".format(i+1,client_loss))
        return {'client_losses':client_losses}

    def server_accuracy(self, algorithm):
        algorithm.server.model.eval()
        accuracy = _accuracy(algorithm.server.model, self.dataloader, self.device)
        if self.verbose:
            print("Server val accuracy: {:.2f}".format(accuracy))
        return {'server_accuracy': accuracy}

    def client_accuracy(self, algorithm):
        client_accuracies = []
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            accuracy = _accuracy(client.model, self.dataloader, self.device)
            client_accuracies.append(accuracy)
            if self.verbose:
                print("Sampled Client {} accuracy: {:.3f}".format(i+1,accuracy))
        return {'client_accuracies':client_accuracies}

    def client_recall(self, algorithm):
        client_recalls = []
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            recall = _recall(client.model, self.dataloader, self.device)
            client_recalls.append(recall)
            if self.verbose:
                print("Sampled Client {} recall: {:.3f}".format(i+1,sum(recall)))
        return {'client_recalls':client_recalls}

    def server_recall(self, algorithm):
        algorithm.server.model.eval()
        recall = _recall(algorithm.server.model, self.dataloader, self.device)
        if self.verbose:
            print("Server average val recall: {:.2f}".format(np.sum(recall) / (len(recall))))
        return {'server_recall': recall}

    def client_precision(self, algorithm):
        client_precisions = []
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            precision = _precision(client.model, self.dataloader, self.device)
            client_precisions.append(precision)
            if self.verbose:
                print("Sampled Client {} precision: {:.3f}".format(i+1,sum(precision)))
        return {'client_precision':client_precisions}

    def server_precision(self, algorithm):
        algorithm.server.model.eval()
        precision = _precision(algorithm.server.model, self.dataloader, self.device)
        if self.verbose:
            print("Server average val precision: {:.2f}".format(np.sum(precision) / (len(precision))))
        return {'server_precision': precision}

    def client_f1(self, algorithm):
        client_f1s =[]
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            f1 = _f1(client.model, self.dataloader, self.device)
            client_f1s.append(f1)
            if self.verbose:
                print("Sampled Client {} f1 score: {:.3f}".format(i+1,sum(f1)))
        return {'client_f1':client_f1s}

    def server_f1(self, algorithm):
        algorithm.server.model.eval()
        f1 = _f1(algorithm.server.model, self.dataloader, self.device)
        if self.verbose:
            print("Server average val f1 score: {:.2f}".format(np.sum(f1) / (len(f1))))
        return {'server_f1': f1}

    def server_training_loss(self, algorithm):
        client_losses = []
        server = algorithm.server
        for i, client in enumerate(algorithm.clients):
            client_loss = server.evaluate(client.dataloader, device = self.device,take_mean=False)
            client_loss = np.sum(client_loss)
            client_losses.append(client_loss)
            if self.verbose:
                print("Server Model on sampled Client {} training loss: {:.3f}".format(i+1,client_loss))
        return {'server_training_loss':client_losses}

    def client_training_loss(self, algorithm):
        client_losses = []
        for i, client in enumerate(algorithm.clients):
            client_loss = client.evaluate(client.dataloader, device = self.device,take_mean=False)
            client_loss = np.sum(client_loss)
            client_losses.append(client_loss)
            if self.verbose:
                print("Sampled Client {} training loss: {:.3f}".format(i+1,client_loss))
        return {'client_training_loss':client_losses}

    def server_training_accuracy(self, algorithm):
        client_accuracies = []
        model = algorithm.server.model
        model.eval()
        for i, client in enumerate(algorithm.clients):
            accuracy = _accuracy(model, client.dataloader, self.device)
            client_accuracies.append(accuracy)
            if self.verbose:
                print("Server Model on sampled Client {} training accuracy: {:.3f}".format(i+1,accuracy))
        return {'server_training_accuracy':client_accuracies}

    def server_training_thesis_results(self, algorithm):
        client_accuracies = []
        client_recall = []
        client_precision = []
        client_loss = []
        model = algorithm.server.model
        model.eval()
        for i, client in enumerate(algorithm.clients):
            (accuracy, recall, precision, loss) = _accrecprec(model, client.dataloader, self.device)
            client_accuracies.append(accuracy)
            client_recall.append(recall)
            client_precision.append(precision)
            clinet_loss.append(loss)
            if self.verbose:
                print("Server Model on sampled Client {} training accuracy: {:.3f}".format(i+1,accuracy))
                print("Server Model on sampled Client {} training recall: {:.3f}".format(i+1,recall))
                print("Server Model on sampled Client {} training precision: {:.3f}".format(i+1,precision))
                print("Server Model on sampled Client {} training loss: {:.4f}".format(i+1,loss))
        return {'server_training_accuracy':client_accuracies, 'server_training_recall': client_recall, 'server_training_precision': client_precision, 'server_training_loss':client_loss}

    def client_training_accuracy(self, algorithm):
        client_accuracies = []
        for i, client in enumerate(algorithm.clients):
            client.model.eval()
            accuracy = _accuracy(client.model, client.dataloader, self.device)
            client_accuracies.append(accuracy)
            if self.verbose:
                print("Sampled Client {} training accuracy: {:.3f}".format(i+1,accuracy))
        return {'client_training_accuracy':client_accuracies}

def _accrecprec(model, dataloader, device):
    loss_function = torch.nn.CrossEntropyLoss()
    output_labels = []
    targets = []
    loss = 0
    for i, (data, target) in enumerate(dataloader):
        targets.extend(target.tolist())
        output = model.predict(data, device = device)
        output = output[0].cpu()
        loss += loss_function(output, target).item()
        n_labels = output[0].shape[-1]
        output_labels.extend(torch.argmax(output, axis = -1).tolist())
    matrix = confusion_matrix(targets, output_labels, labels = np.arange(n_labels))
    true_pos = np.diag(matrix)
    precision = np.mean(true_pos / np.sum(matrix, axis = 0))
    recall = np.mean(true_pos / np.sum(matrix, axis = 1))
    accuracy = np.sum(true_pos) / np.sum(matrix)
    return (accuracy, recall, precision, loss)

def _model_weight_to_array(w):
    flattened = torch.cat([w[k].flatten() for k in w]).detach()
    return flattened

def _accuracy(model, dataloader, device):
    acc = 0
    for data, target in dataloader:
        output = model.predict(data, device = device)
        output_labels = torch.argmax(output[0], axis = -1).to('cpu')
        acc += torch.sum(output_labels == target)
    return acc.item() / len(dataloader.dataset)

def _recall(model, dataloader, device):
    target_true = torch.Tensor()
    target_pred = torch.Tensor()
    for data, target in dataloader:
        output = model.predict(data, device = device)
        output_labels = torch.argmax(output[0], axis = -1).to('cpu')
        target_true = torch.cat((target_true, target))
        target_pred = torch.cat((target_pred, output_labels))
    return list(recall_score(target_true, target_pred, average = None))

def _precision(model, dataloader, device):
    target_true = torch.Tensor()
    target_pred = torch.Tensor()
    for data, target in dataloader:
        output = model.predict(data, device = device)
        output_labels = torch.argmax(output[0], axis = -1).to('cpu')
        target_true = torch.cat((target_true, target))
        target_pred = torch.cat((target_pred, output_labels))
    return list(precision_score(target_true, target_pred, average = None))

def _f1(model, dataloader, device):
    target_true = torch.Tensor()
    target_pred = torch.Tensor()
    for data, target in dataloader:
        output = model.predict(data, device = device)
        output_labels = torch.argmax(output[0], axis = -1).to('cpu')
        target_true = torch.cat((target_true, target))
        target_pred = torch.cat((target_pred, output_labels))
    return list(f1_score(target_true, target_pred, average = None))
