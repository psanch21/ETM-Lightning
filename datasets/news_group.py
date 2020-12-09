import os
import pickle

import numpy as np
import scipy.io
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets.corpus_dataset import CorpusDataset


def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts,
                'tokens_1': tokens_1, 'counts_1': counts_1,
                'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}


def get_data(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')

    return vocab, train, valid, test


class NewsGroupDataModule(LightningDataModule):
    name = "20ng"

    def __init__(
            self,
            data_dir: str = "./",
            num_workers: int = 16,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            num_workers: how many workers to use for loading data
            batch_size: size of batch
        """
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.vocab, train, valid, test = get_data(os.path.join(data_dir))

        self.dataset_dict = {}
        self.dataset_dict['train'], self.n_tr = self.create_dataset(train)
        self.dataset_dict['valid'], self.n_val = self.create_dataset(valid)
        self.dataset_dict['test'], self.n_tst = self.create_dataset(test)
        self.info()

    @property
    def vocab_size(self):
        """
        Return:
            784
        """
        return len(self.vocab)

    def info(self):
        print(f"\nVocab size: {self.vocab_size}")
        print(f"\nTraining samples: {self.n_tr}")
        print(f"Validation samples: {self.n_val}")
        print(f"Test samples: {self.n_tst}\n")

    def create_dataset(self, corpus_data):
        tokens = corpus_data['tokens']
        counts = corpus_data['counts']
        batch_size = len(tokens)
        data_batch = np.zeros((batch_size, self.vocab_size))
        for i, (doc, count) in enumerate(zip(tokens, counts)):
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
        dataset = CorpusDataset(data_batch)
        return dataset, batch_size

    def train_dataloader(self):

        loader = DataLoader(
            self.dataset_dict['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):

        loader = DataLoader(
            self.dataset_dict['valid'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_dict['test'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
