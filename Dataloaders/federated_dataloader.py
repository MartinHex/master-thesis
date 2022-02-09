from abc import ABC, abstractmethod

class FederatedDataLoader(ABC):

    @abstractmethod
    def get_training_dataloaders(self):
        pass

    @abstractmethod
    def get_test_dataloader(self):
        pass

    @abstractmethod
    def get_training_raw_data(self):
        pass

    @abstractmethod
    def get_test_raw_data(self):
        pass
