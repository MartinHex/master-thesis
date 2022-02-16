# Dataloaders
For our thesis we have created wrappers that return lists of PyTorch Dataloaders that can be used as local datasets for each client. In general each wrapper support the following methods:
```python
get_training_raw_data()
get_training_dataloaders(batch_size, shuffle = True)
get_test_raw_data()
get_test_dataloader(batch_size)
```
The 'raw' data is data on the form of a list with tuples containing (Tensor, target). The data for each client is the same for both Dataloader and in the list form, the only reason to include the list-form is to make visualization and analysis easier.

## MNIST

The MNIST dataset is loaded using  `torchvision.datasets.MNIST()`, as such the test and train partitions are those given by torchvision. The number of clients selected need to be evenly dividable with the number of samples in the training partition, the guarantees that there are the same number of samples on each client. The dataset is split into i.i.d. clients.

## CIFAR100

The CIFAR-100 dataset is loaded using `torchvision.datasets.CIFAR100`, as such the test and train aprtitions are those given by torchvision. The number of clients selected need to be evenly dividable with the number of smaples in the training partition, this guarantees that there are the same number of samples on each clients.

To split the  dataset into clients a two step latent Dirichlet allocation is used, as proposed by [S. Reddit et al.](https://arxiv.org/pdf/2003.00295.pdf) The CIFAR-100 dataset is split into 10 coarse labels that each is related to 10 fine labels. The intuitive reason for a two step LDA distribution is that clients that observe one coarse label is more likely to observe multiple of the fine labels included in that coarse label. For example, a client that observe the coarse label 'mammals' is more likely to also observe many of the fine labels that correspond to said coarse label rather than fine labels of other coarse labels, meaning each clients is somewhat limited to a domain. This generate a more natural split of clients.

## EMNIST-62

The EMNIST-62, aka FEMNIST, dataset is created by [S. Caldas et al.](https://arxiv.org/pdf/1812.01097.pdf) and is downloaded through the [LEAF](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist) repository. The current dataloader wrapper support splitting by selecting a number of authors randomly, as such the authors are selected i.i.d. but the class labels will be non-i.i.d. as each author contain different amount of each label.

Currently the dataloader loads data by accessing the `all_data` directory created by executing the `data_to_json.sh` script supplied by the LEAF repository. At the moment downloading the data must be done manually and the dataloader assume access to the required JSON-files.

TODO: Make the download of EMNIST-62 more dynamic such that it download to the correct directory automatically.

## Stack Overflow

TODO
