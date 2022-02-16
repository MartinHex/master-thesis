from Dataloaders.federated_dataloader import FederatedDataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

class EMNIST(FederatedDataLoader):
    """Federated wrapper class for the Torchvision EMNIST-62 dataset

    Assumes:
        - That the Federated EMNIST-62 (FEMNIST) dataset is present in ../data/femnist.
        - The data is split by using the method presented by Caldas et al. (https://github.com/TalwalkarLab/leaf)
    """
    def __init__(self, number_of_clients, test_size = 0.3, data_path = None, test_path = None):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = os.path.join('..', 'data', 'femnist' 'all_data')

        self.number_of_clients = number_of_clients
        self.test_size = int(self.number_of_clients * test_size)

        assert self.test_size > 0, "The number of test clients is less than 1"

        self.testset = []
        self.trainset = []

        self._test_train_split()
        self.unified_testset = [item for sublist in self.testset for item in sublist]

    def get_training_dataloaders(self, batch_size, shuffle = True):
        dataloaders = []
        for client in self.trainset:
            dataloaders.append(DataLoader(ImageDataset(client), batch_size = batch_size, shuffle = shuffle))
        return dataloaders

    def get_test_dataloader(self, batch_size):
        return DataLoader(ImageDataset(self.unified_testset), batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.trainset

    def get_test_raw_data(self):
        return self.unified_testset

    def _test_train_split(self):
        all_clients = self._get_all_clients()
        clients = list(all_clients.keys())

        train_index = np.random.choice(range(len(clients)), size = self.number_of_clients, replace = False)
        train_clients = [clients[i] for i in train_index]#clients[train_index]

        test_index = np.random.choice(list(set(range(len(clients))) - set(train_index)), size = self.test_size, replace = False)
        test_clients = [clients[i] for i in test_index] #clients[test_index]

        assert len(set(train_clients) | set(test_clients)) == (len(set(train_clients)) + len(set(test_clients))), "Clients appear in both training and test set"

        print('Loading training clients...')
        for client in tqdm(train_clients):
            file = all_clients[client]
            self.trainset.append(self._read_client(client, file))
            print('Loading test clients...')
        for client in tqdm(test_clients):
            file = all_clients[client]
            self.testset.append(self._read_client(client, file))

    def _read_client(self, client, file):
        file_path = os.path.join(self.data_path, file)
        with open(file_path, 'r') as inf:
            data = json.load(inf)

        client_data = list(zip(data['user_data'][client]['x'], data['user_data'][client]['y']))
        for i, (x, y) in enumerate(client_data):
            client_data[i] = (torch.reshape(torch.Tensor(x), (28,28)), y)

        return client_data

    def _get_all_clients(self):
        all_clients = dict()
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('.json')]
        print('Collecting all available authors...')
        for f in tqdm(files):
            file_path = os.path.join(self.data_path,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            file_clients = cdata['users']
            for client in file_clients:
                all_clients[client] = f
        return all_clients

class ImageDataset(Dataset):
    """Constructor

    Args:
        data: list of data for the dataset.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        """Get sample by index

        Args:
            index (int)

        Returns:
             The index'th sample (Tensor, int)
        """
        tensor, label = self.data[index]
        return tensor, label

    def __len__(self):
        """Total number of samples"""
        return len(self.data)
