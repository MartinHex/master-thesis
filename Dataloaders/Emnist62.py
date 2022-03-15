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
    def __init__(self, number_of_clients, test_size = 0.3, data_path = None, test_path = None, seed = 1234, client_threshold = 150):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = os.path.join('data', 'femnist', 'all_data')

        np.random.seed(seed = seed)
        self.test_size = test_size
        self.client_size_threshold = client_threshold

        self.testset = []
        self.trainset = []

        self._test_train_split()

    def get_training_dataloaders(self, batch_size, shuffle = True):
        dataloaders = []
        for client in self.trainset:
            dataloaders.append(DataLoader(ImageDataset(client), batch_size = batch_size, shuffle = shuffle))
        return dataloaders

    def get_test_dataloader(self, batch_size):
        return DataLoader(ImageDataset(self.testset), batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.trainset

    def get_test_raw_data(self):
        return self.testset

    def _test_train_split(self):
        all_clients = self._get_all_clients()
        for client in all_clients:
            if (len(client) >= self.client_size_threshold):
                test_size = int(self.test_size * len(client))
                train_index = np.random.choice(range(len(client)), size = (len(client) - test_size), replace = False)
                test_index = np.random.choice(list(set(range(len(client))) - set(train_index)), size = test_size, replace = False)
                assert len(set(train_index) | set(test_index)) == (len(set(train_index)) + len(set(test_index))), "Clients appear in both training and test set"

                train_data = [client[i] for i in train_index]
                test_data = [client[i] for i in test_index]
                if len(train_data) > 0: self.trainset.append(train_data)
                if len(test_data) > 0:self.testset.extend(test_data)

    def _get_all_clients(self):
        all_clients = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('.json')]
        print('Collecting all available authors...')
        for f in tqdm(files):
            file_path = os.path.join(self.data_path,f)
            with open(file_path, 'r') as data:
                data = json.load(data)
            file_clients = data['users']
            for client in file_clients:
                client_data = list(zip(data['user_data'][client]['x'], data['user_data'][client]['y']))
                for i, (x, y) in enumerate(client_data):
                    client_data[i] = (torch.reshape(torch.Tensor(x), (1, 28, 28)), y)
                all_clients.append(client_data)
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
