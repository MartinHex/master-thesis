from Dataloaders.federated_dataloader import FederatedDataLoader
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Mnist(FederatedDataLoader):
    """Federated wrapper class for the Torchvision MNIST dataset

    The class splits the MNIST training data into the desired amount of client with a uniform distribution.
    """
    def __init__(self, number_of_clients,  download = True):
        """Constructor

        Args:
            number_of_clients: how many federated clients to split the data into (int).
            download: whether to allow torchvision to download the dataset (default: True)

        """
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()
            ])

        self.number_of_clients = number_of_clients

        self.trainset = torchvision.datasets.MNIST(root= './data', train = True, download = download, transform = self.transform)

        assert len(self.trainset) % self.number_of_clients == 0, "Number of clients must be evenly devicible with the length of the dataset, length of the dataset is {}".format(len(self.trainset))

        self.testset = [(x, y) for x, y in torchvision.datasets.MNIST(root = './data', train = False, download = download, transform = self.transform)]

        self.split_trainset = self._create_trainset()

    def get_training_dataloaders(self, batch_size, shuffle = True):
        dataloaders = []
        for client in self.split_trainset:
            dataloaders.append(DataLoader(ImageDataset(client), batch_size = batch_size, shuffle = shuffle))
        return dataloaders

    def get_test_dataloader(self, batch_size):
        return DataLoader(ImageDataset(self.testset), batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.split_trainset

    def get_test_raw_data(self):
        return self.testset

    # Split the dataset into clients, if we want to make the dataset non-iid this is where we'd do that.
    def _create_trainset(self):
        randomized_index = np.random.choice(len(self.trainset), len(self.trainset), replace = False)
        client_list = [[] for i in range(self.number_of_clients)]
        for i in range(len(self.trainset)):
            client_index = int( (i * self.number_of_clients) / len(self.trainset))
            client_list[client_index].append((self.trainset[randomized_index[i]][0], self.trainset[randomized_index[i]][1]))
        return client_list

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
