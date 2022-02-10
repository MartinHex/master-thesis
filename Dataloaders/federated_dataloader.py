from abc import ABC, abstractmethod

class FederatedDataLoader(ABC):

    @abstractmethod
    def get_training_dataloaders(self, batch_size, shuffle = True):
        """Get a list of dataloaders containing partitioned training data from the MNIST dataset

        Args:
            batch_size: batch_size for the dataloaders (int).
            shuffle: whether to shuffle the data (Boolean).
        Returns:
            List of dataloaders with data in format (Tensor, int).
        """
        pass

    @abstractmethod
    def get_test_dataloader(self, batch_size):
        """Get a dataloader containing training data from the MNIST dataset

        Args:
            batch_size: batch_size for the dataloaders (int).
        Returns:
            Dataloader with data in format (Tensor, int).
        """
        pass

    @abstractmethod
    def get_training_raw_data(self):
        """Get the training data in the same partition as in the dataloaders but in form of lists of Tensors

        Returns:
            List of list with each list containing the data for one client on the format (Tensor, int)
        """
        pass

    @abstractmethod
    def get_test_raw_data(self):
        """Get the test data in the form of a list of Tensors

        Returns:
            List of the test data on the format (Tensor, int)
        """
        pass
