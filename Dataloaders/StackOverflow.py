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
    def __init__(self, number_of_clients, n_words=20,n_entries=128,seed=0,vocab_size=10000,test_size=0.25):
        """Constructor

        Args:
            number_of_clients: how many federated clients to split the data into (int).
            n_words: number of words used per user entry.
            n_entries: Number of entries per user.
            seed: random seed for sampling clients.
        """
        root = os.path.join('data')
        data_path = os.path.join(root,'Stackoverflow')
        usr_cmnts_pth = os.path.join(data_path,'userEntries_%i_%i.json'%(n_entries,n_words))
        self.sequence_length=20
        self.number_of_clients = number_of_clients

        # Download if not new data
        if(not os.path.exists(usr_cmnts_pth)):
            downloadStackOverflow(root=data_path,n_entries=n_entries,n_clients=10000)

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
        random_sample = random.sample(list(usr_cmnts),number_of_clients)

        # Add placeholder indicies to data.
        self.vocab.add('OoV')
        self.vocab.add('BoS')
        self.vocab.add('EoS')
        self.vocab.add('Pad')

        # Preproccess data for the current vocab
        for usr in random_sample:
            for usr_cmnt in usr_cmnts[usr]:
                usr_cmnt['tokens'] = self.preprocess(usr_cmnt['tokens'],min_words=n_words)

        # Construct training and test data
        sequences = [[cmnt['tokens'] for cmnt in usr_cmnts[usr]]
                            for usr in random_sample]
        # Split into 2 different sets
        test_size = int(n_entries*test_size)
        self.split_trainset = [seq[test_size:] for seq in sequences ]
        self.testset = [seq[:test_size] for seq in sequences]
        # Flatten testset
        self.testset = [seq for usr_seq in self.testset for seq in usr_seq]


    def get_training_dataloaders(self, batch_size, shuffle = True):
        dataloaders = []
        for client in self.split_trainset:
            dataset = WordSequenceDataset(sequences=client,vocab=self.vocab,
                                            sequence_length=self.sequence_length)
            dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = shuffle)
            dataloaders.append(dataloader)
        return dataloaders

    def get_test_dataloader(self, batch_size):
        dataset = WordSequenceDataset(sequences=self.testset,vocab=self.vocab,
                                        sequence_length=self.sequence_length)
        return DataLoader(dataset,batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.split_trainset

    def get_test_raw_data(self):
        return self.testset

    def get_vocab(self):
        return self.vocab

    def preprocess(self,sentence,min_words=20):
        for i,w in enumerate(sentence):
            if(w not in self.vocab):
                sentence[i] = 'OoV'
        res_seq = ['BoS']+sentence+['EoS']
        remaining_space = min_words-len(res_seq)+1
        if(remaining_space>0):
            res_seq= ['Pad']*remaining_space+res_seq
        return res_seq


def downloadStackOverflow(root='./',n_entries=128,n_clients=10000):
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

    res = {}
    for user_id in train.client_ids:
      user_entries = tfds.as_numpy(train.create_tf_dataset_for_client(user_id))
      usr_data = [{} for _ in range(n_entries)]
      ix = 0
      for entry in user_entries:
        text = entry['tokens'].decode("utf-8")
        text = ''.join([t for t in text if t.isalpha() or t==" "])
        text = text.split(" ")
        entry['tokens'] = [t for t in text if t!='']
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

      # If gathered enough clients, save results and finish run.
      if(len(list(res))==n_clients):
          outputpath = os.path.join(root,'userEntries_%i_%i.json'%(n_entries,n_words))
          with open(outputpath,'w', encoding='utf-8') as f:
              json.dump(res, f,ensure_ascii=False)
          break

class WordSequenceDataset(Dataset):
    """Constructor

    Args:
        data: list of data for the dataset.
    """
    def __init__(self,sequences,vocab,sequence_length=20):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.vocab = vocab
        # Set up indexing for dynamically loading different subsets of sentences.
        samples_per_sentence = [len(s)-sequence_length for s in sequences]
        self.indexing = [[i,j] for i in range(len(sequences)) for j in range(samples_per_sentence[i])]

        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

    def __len__(self):
        return len(self.indexing)

    def __getitem__(self, index):
        seq,first_word = self.indexing[index]
        sequence = self.sequences[seq][first_word:(first_word+self.sequence_length+1)]
        sequence = [self.word_to_index[w] for w in sequence]
        return (
            tensor(sequence[:-1]),
            tensor(sequence[1:])
        )
