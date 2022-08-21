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
    def __init__(self, number_of_clients = 10000, n_words=20,n_entries=128,seed=0,vocab_size=10000,test_size=0.01):
        """Constructor
        Args:
            number_of_clients: how many federated clients to split the data into (int).
            n_words: number of words used per user entry.
            n_entries: Number of entries per user.
            seed: random seed for sampling clients.
        """
        random.seed(seed)
        root = os.path.join('data')
        if not os.path.exists(root): os.makedir(root)
        data_path = os.path.join(root,'Stackoverflow')
        if not os.path.exists(data_path): os.mkdir(data_path)
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

        # Add placeholder indicies to data.
        self.vocab.add('OoV')
        self.vocab.add('BoS')
        self.vocab.add('EoS')
        self.vocab.add('Pad')

        # Set up maps
        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        # Sample clients
        random_sample = random.sample(list(usr_cmnts),number_of_clients)

        # Set up raw data
        sequences = [[cmnt['tokens'] for cmnt in usr_cmnts[usr]]
                            for usr in random_sample]
        test_size = int(n_entries*test_size)
        self.raw_trainset = [seq[test_size:] for seq in sequences ]
        self.raw_testset = [seq[:test_size] for seq in sequences]

        # Preproccess data
        self.trainset = self.preprocess(self.raw_trainset)
        self.testset = self.preprocess(self.raw_testset)

        # Flatten testset
        self.testset = [seq for usr_seq in self.testset for seq in usr_seq]
        self.raw_testset = [seq for usr_seq in self.raw_testset for seq in usr_seq][0]


    def get_training_dataloaders(self, batch_size, shuffle = True):
        dataloaders = []
        non_idx = (
            self.word_to_index['OoV'],
            self.word_to_index['Pad'],
            self.word_to_index['BoS'],
        )
        for client in self.trainset:
            dataset = WordSequenceDataset(sequences=client,n_words=20,non_idx=non_idx)
            dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = shuffle)
            dataloaders.append(dataloader)
        return dataloaders

    def get_test_dataloader(self, batch_size):
        non_idx = (
            self.word_to_index['OoV'],
            self.word_to_index['Pad'],
            self.word_to_index['BoS'],
        )
        dataset = WordSequenceDataset(sequences=self.testset,n_words=20,non_idx=non_idx)
        return DataLoader(dataset,batch_size = batch_size, shuffle = False)

    def get_training_raw_data(self):
        return self.raw_trainset

    def get_test_raw_data(self):
        return self.raw_testset

    def get_vocab(self):
        return self.vocab

    def preprocess(self,users,n_words=20):
        res = []
        for usr in users:
            usr_res = []
            for cmnt in usr:
                preprocessed_cmnt = self.preprocess_cmt(cmnt,n_words=n_words)
                usr_res.append(preprocessed_cmnt)
            res +=[usr_res]
        return res

    def preprocess_cmt(self,cmnt,n_words):
        res_seq = cmnt.copy()
        for i,w in enumerate(cmnt):
            if(w not in self.vocab):
                res_seq[i] = 'OoV'
        res_seq = (n_words-1)*['Pad']+['BoS']+res_seq+['EoS']
        res_seq = [self.word_to_index[w]for w in res_seq]
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
    def __init__(self,sequences,n_words=20,non_idx=None):
        self.n_words = n_words
        self.sequences = sequences
        # Set up indexing for dynamically loading different subsets of sentences.
        self.indexing = []
        for i,seq in enumerate(sequences):
            target_indecis = []
            for j,word in enumerate(seq):
                if word not in non_idx: self.indexing += [(i, j)]

    def __len__(self):
        return len(self.indexing)

    def __getitem__(self, index):
        seq,last_word = self.indexing[index]
        sequence = self.sequences[seq][(last_word-self.n_words):(last_word + 1)]
        return (
            tensor(sequence[:-1]),
            tensor(sequence[1:])
        )
