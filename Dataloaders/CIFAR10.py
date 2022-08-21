from Dataloaders.federated_dataloader import FederatedDataLoader
import torchvision
import numpy as np
import torch
import torchvision.transforms as tt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict

class CIFAR10(FederatedDataLoader):
    """Federated wrapper class for the Torchvision DirichletCifar100 dataset
    The wrapper trim away data the minimum amount of data needed to make the dataset IID w.r.t. labels and such that it can be split into the number of clients specified.
    The class splits the MNIST training data into the desired amount of client with a uniform distribution as default.
    To split the data in a non-IID fashion tune the alpha parameter to do a LDA with dirichlet parameter alpha.
    """
    def __init__(self, number_of_clients,  download = True, alpha = 'inf', seed = 1234):
        """Constructor
        Args:
            number_of_clients: how many federated clients to split the data into (int).
            download: whether to allow torchvision to download the dataset (default: True)
            alpha: dirichlet parameter for LDA, default is 'inf' indicating IID split (int)
        """
        # Precalculated means and variance of training data
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
        self.transform = tt.Compose(
            [tt.ToTensor(),
            tt.Normalize(mean, std)
            ])
        self.alpha = alpha
        np.random.seed(seed = seed)

        self.number_of_clients = number_of_clients

        trainset = torchvision.datasets.CIFAR10(root= './data', train = True, download = download, transform = self.transform)

        testset = [(x, y) for x, y in torchvision.datasets.CIFAR10(root = './data', train = False, download = download, transform = self.transform)]

        self.mapped_trainset = self._map_set(trainset)
        for key in self.mapped_trainset:
            np.random.shuffle(self.mapped_trainset[key])

        self.mapped_testset = self._map_set(testset)
        for key in self.mapped_testset:
            np.random.shuffle(self.mapped_testset[key])

        self.train_data_amount = self._normalize_data()

        self.testset = []
        for key in self.mapped_testset:
            self.testset.extend([(t,key) for t in self.mapped_testset[key]])
        np.random.shuffle(self.testset)

        self.split_trainset = self._create_trainset()

    def get_training_dataloaders(self, batch_size, shuffle = True):
        """Get a list of dataloaders containing partitioned training data from the CIFAR10 dataset
        Args:
            batch_size: batch_size for the dataloaders (int).
            shuffle: whether to shuffle the data (Boolean).
        Returns:
            List of dataloaders with data in format (Tensor, int).
        """
        datasets = []
        transform = tt.Compose([tt.RandomCrop(24),
                        tt.RandomHorizontalFlip(p=0.5)])
        for client in self.split_trainset:
            datasets.append(ImageDataset(client, transform = transform))
        dataloaders = []
        for dataset in datasets:
            dataloaders.append(DataLoader(dataset, batch_size = batch_size, shuffle = shuffle))

        return dataloaders

    def get_test_dataloader(self, batch_size):
        """Get a dataloader containing training data from the CIFAR10 dataset
        Args:
            batch_size: batch_size for the dataloaders (int).
            label: whether to use the coarse or fine label in the CIFAR100 dataset (fine/coarse)
        Returns:
            Dataloader with data in format (Tensor, int).
        """
        return DataLoader(ImageDataset(self.testset), batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.split_trainset

    def get_test_raw_data(self):
        return self.testset

    # Split the dataset into clients, if we want to make the dataset non-iid this is where we'd do that.
    def _create_trainset(self):
        client_list = [[] for i in range(self.number_of_clients)]
        if self.alpha == 'inf':
            for client in client_list:
                for label in self.mapped_trainset:
                    for i in range(self.train_data_amount // (10 * self.number_of_clients)):
                        tensor = self.mapped_trainset[label].pop()
                        client.append((tensor, label))
        else:
            available_labels = [x for x in range(10)]
            number_of_samples = self.train_data_amount // self.number_of_clients
            for client in range(self.number_of_clients):
                theta = np.random.dirichlet([self.alpha] * len(available_labels))
                for i in range(number_of_samples):
                    index = np.nonzero(np.random.multinomial(1, theta, size = 1))[1][0]
                    label = available_labels[index]
                    tensor = self.mapped_trainset[label].pop()
                    client_list[client].append((
                        tensor,
                        label,
                    ))
                    if (len(self.mapped_trainset[label]) == 0):
                        available_labels.remove(label)
                        theta = np.delete(theta, index)
                        self._renormalize(theta)
        return client_list

    def _map_set(self, data):
        mapped_set = defaultdict(lambda: [])
        for (tensor, label) in data:
            mapped_set[label].append(tensor)
        return mapped_set

    def _renormalize(self, theta):
        s = 0
        for l in theta:
            s += l
        map(lambda x: x/s, theta)

    def _normalize_data(self):
        min_test_frequency = 100000000000
        for key in self.mapped_testset:
            if len(self.mapped_testset[key]) < min_test_frequency: min_test_frequency = len(self.mapped_testset[key])
        for key in self.mapped_testset:
            self.mapped_testset[key] = self.mapped_testset[key][0:min_test_frequency]

        min_train_frequency = 100000000000
        for key in self.mapped_trainset:
            if len(self.mapped_trainset[key]) < min_train_frequency: min_train_frequency = len(self.mapped_trainset[key])
        train_class_size = ((min_train_frequency // self.number_of_clients) * self.number_of_clients)
        for key in self.mapped_trainset:
            self.mapped_testset[key] = self.mapped_testset[key][0:train_class_size]
        return train_class_size * 10

    def _create_trainset(self):
        client_list = [[] for i in range(self.number_of_clients)]
        if self.alpha == 'inf':
            for client in client_list:
                for label in self.mapped_trainset:
                    for i in range(self.train_data_amount // (10 * self.number_of_clients)):
                        tensor = self.mapped_trainset[label].pop()
                        client.append((tensor, label))
        else:
            available_labels = [x for x in range(10)]
            number_of_samples = self.train_data_amount // self.number_of_clients
            for client in range(self.number_of_clients):
                theta = np.random.dirichlet([self.alpha] * len(available_labels))
                for i in range(number_of_samples):
                    index = np.nonzero(np.random.multinomial(1, theta, size = 1))[1][0]
                    label = available_labels[index]
                    tensor = self.mapped_trainset[label].pop()
                    client_list[client].append((
                        tensor,
                        label,
                    ))
                    if (len(self.mapped_trainset[label]) == 0):
                        available_labels.remove(label)
                        theta = np.delete(theta, index)
                        self._renormalize(theta)
        return client_list


class ImageDataset(Dataset):
    """Constructor
    Args:
        data: list of data for the dataset.
    """
    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        """Get sample by index
        Args:
            index (int)
        Returns:
             The index'th sample (Tensor, int)
        """
        tensor, label = self.data[index]
        if self.transform != None: tensor = self.transform(tensor)
        return tensor, label

    def __len__(self):
        """Total number of samples"""
        return len(self.data)
