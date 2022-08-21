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
    def __init__(self, train_path = None, test_path = None):
        if train_path:
            self.train_path = train_path
            self.test_path = test_path
        else:
            self.train_path = os.path.join('data', 'femnist', 'train')
            self.test_path = os.path.join('data', 'femnist', 'test')

        self.testset = self._get_clients(self.test_path)
        self.trainset = self._get_clients(self.train_path)
        self.testset = [item for sublist in self.testset for item in sublist]

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

    def _get_clients(self, path):
        all_clients = []
        files = os.listdir(path)
        files = [f for f in files if f.endswith('.json')]
        print('Collecting all available authors on path: {}'.format(path))
        for f in tqdm(files):
            file_path = os.path.join(path,f)
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
