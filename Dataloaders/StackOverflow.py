from Dataloaders.federated_dataloader import FederatedDataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
import json
import os
import random
from collections import defaultdict
import numpy as np


class StackOverflow(FederatedDataLoader):
    """Federated wrapper class for the Tensorflow Stack Overflow dataset

    The class splits the StackOverflow in seperate dataloaders for each seperated user.
    Doing so it has to download Stack Overflow data from Tensorflow-Federated.
    This requires some time and to stop this from being a requirement, resulting
    preprocessed data is stored in a data folder.
    This data in json format userComments_{n_entries}_{n_words}.json .
    """
    def __init__(self, number_of_clients, n_words=20,n_entries=128,seed=0,vocab_size=10000):
        """Constructor

        Args:
            number_of_clients: how many federated clients to split the data into (int).
            n_words: number of words used per user entry.
            n_entries: Number of entries per user.
            seed: random seed for sampling clients.
        """
        root = os.path.join('data')
        data_path = os.path.join(root,'Stackoverflow')
        usr_cmnts_pth = os.path.join(data_path,'userComments_%i_%i.json'%(n_entries,n_words))

        self.number_of_clients = number_of_clients

        # Download if not new data
        if(not os.path.exists(usr_cmnts_pth)):
            downloadStackOverflow(root=data_path,n_entries=n_entries,n_words=n_words,n_clients=10000)

        # Load data
        with open(usr_cmnts_pth) as f:
            usr_cmnts = json.load(f)

        # Get vocabolary
        # Using default dict we only iterate through the data once
        counts = defaultdict(int)
        for usr in usr_cmnts:
            for usr_cmnt in usr_cmnts[usr]:
                for word in usr_cmnt['tokens']:
                    counts[word]+=1
        counts_srtd = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
        self.vocab = set(list(counts_srtd)[-vocab_size:])

        # Sample train and testing data
        random_sample = random.sample(list(usr_cmnts),2*number_of_clients)

        # Preproccess data for the current vocab
        for usr in random_sample:
            for usr_cmnt in usr_cmnts[usr]:
                for i,word in enumerate(usr_cmnt['tokens']):
                    if(word not in self.vocab):
                        usr_cmnt['tokens'][i]='Out_Of_Vocab'

        self.vocab.add('Out_Of_Vocab')
        # Construct training and test data
        sequences = [[cmnt['tokens'] for cmnt in usr_cmnts[usr]]
                            for usr in random_sample]
        self.split_trainset = sequences[:number_of_clients]
        self.testset = [seq for usr_seq in  sequences[(number_of_clients+1):]
                        for seq in usr_seq]


    def get_training_dataloaders(self, batch_size, shuffle = True):
        dataloaders = []
        for client in self.split_trainset:
            dataset = WordSequenceDataset(sequences=client,vocab=self.vocab)
            dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = shuffle)
            dataloaders.append(dataloader)
        return dataloaders

    def get_test_dataloader(self, batch_size):
        dataset = WordSequenceDataset(sequences=self.testset,vocab=self.vocab)
        return DataLoader(dataset,batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.split_trainset

    def get_test_raw_data(self):
        return self.testset

    def get_vocab(self):
        return self.vocab



def downloadStackOverflow(root='./',n_entries=128,n_words=20,n_clients=10000):
    """
        Downloads stackoverflow data through tensorflow_federated.
        This data is then translated into a pytorch formal.

    """
    print("Downloading and translation of StackOverflow may take a while.")
    print("Observe that TFF is only supported on linux systems.")
    import tensorflow_federated as tff
    import tensorflow_datasets as tfds

    # Load data
    (train, held_out, test) = tff.simulation.datasets.stackoverflow.load_data(
        cache_dir=None
    )
    n_entries = 128
    n_words = 20
    res = {}
    for user_id in train.client_ids:
      user_entries = tfds.as_numpy(train.create_tf_dataset_for_client(user_id))
      usr_data = [{} for _ in range(n_entries)]
      ix = 0
      for entry in user_entries:
        text = entry['tokens'].decode("utf-8")
        text = ''.join([t for t in text if t.isalpha() or t==" "])
        text = text.split(" ")
        entry['tokens'] = [t for t in text if t!=''][:n_words]
        if(len(entry['tokens'])==20):
          entry['creation_date'] = entry['creation_date'].decode("utf-8")
          entry['tags'] = entry['tags'].decode("utf-8")
          entry['type'] = entry['type'].decode("utf-8")
          entry['title'] = entry['title'].decode("utf-8")
          entry['score'] = int(entry['score'])
          usr_data[ix] = dict(entry)
          if(ix+1==n_entries):
            res[user_id] = usr_data
            break
          ix+=1
      if(len(list(res))==n_clients):
          outputpath = os.path.join(root,'userComments_%i_%i.json'%(n_entries,n_words))
          with open(outputpath,'w', encoding='utf-8') as f:
              json.dump(res, f,ensure_ascii=False)
          break

class WordSequenceDataset(Dataset):
    """Constructor

    Args:
        data: list of data for the dataset.
    """
    def __init__(self,sequences,vocab=[],padding=3):
        self.vocab = vocab
        self.sequences = sequences

        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = [self.word_to_index[w] for w in self.sequences[index]]
        return (
            tensor(sequence[:-1]),
            tensor(sequence[1:])
        )
