
import torch
from torch.utils.data import Dataset
class CorpusDataset(Dataset):
    """Twitter dataset."""

    def __init__(self, corpus):
        """
        Args:
            corpus (np.array): NxI array
        """
        self.corpus = torch.from_numpy(corpus).float()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.corpus[idx]
