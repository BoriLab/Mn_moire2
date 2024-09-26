# %%
from torch.utils.data import DataLoader, Dataset
import random
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_dataloader(self, split="train"):
    if split == "train":
        sampler = SubsetRandomSampler(self.indices[split])
        shuffle = False  # Sampler handles the shuffling
    else:
        sampler = self.indices[split]  # Use these indices as they are
        shuffle = False
    return DataLoader(
        self,
        batch_size=self.batch_size,
        sampler=sampler,
        collate_fn=collate_batch,
        shuffle=shuffle,  # Sampler makes shuffle unnecessary
    )


def collate_batch(batch):
    node_features, adj_matrices, targets = zip(*batch)
    max_nodes = max([nf.size(0) for nf in node_features])
    node_masks = [
        torch.cat([torch.ones(nf.size(0)), torch.zeros(max_nodes - nf.size(0))])
        for nf in node_features
    ]
    padded_node_features = [
        F.pad(nf, (0, 0, 0, max_nodes - nf.size(0))) for nf in node_features
    ]
    padded_adj_matrices = [
        F.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(1)))
        for adj in adj_matrices
    ]
    padded_adj_matrices = [
        F.pad(
            adj,
            (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(1)),
            value=1e6,
        )
        for adj in adj_matrices
    ]
    node_masks = torch.stack(node_masks).bool()
    node_features = torch.stack(padded_node_features)
    adj_matrices = torch.stack(padded_adj_matrices)
    targets = torch.stack(targets)
    return (node_features, adj_matrices, node_masks, targets)


class MyDataset(Dataset):
    def __init__(
        self, *features, evaluation_size=0.025, test_size=0.025, batch_size=32
    ):
        node_features = features[0]
        # NOTE: This is a list of tuple features
        # * adj_matrices = features[1]
        # * masks = features[2]
        # * targets = features[-1]
        for feature in features:
            assert len(feature) == len(node_features)
        for i, feature in enumerate(features):
            for j, f in enumerate(feature):
                if sp.issparse(f):
                    features[i][j] = f.toarray()
                features[i][j] = torch.tensor(features[i][j])
        indices = list(range(len(node_features)))
        random.shuffle(indices)
        if evaluation_size < 1:
            evaluation_size = int(evaluation_size * len(node_features))
        if test_size < 1:
            test_size = int(test_size * len(node_features))
        self.indices = {
            "train": indices[test_size + evaluation_size :],
            "eval": indices[:evaluation_size],
            "test": indices[evaluation_size : test_size + evaluation_size],
        }
        self.features = features
        self.batch_size = batch_size
        self.node_feat_size = node_features[0].shape[1]
        self.prediction_size = 0
        try:
            self.prediction_size = features[-1][0].shape[1]
        except:
            self.prediction_size = 1

    def float(self):
        for i, feature in enumerate(self.features):
            for j, f in enumerate(feature):
                self.features[i][j] = f.float()

    def unsqueeze_target(self):
        for i, target in enumerate(self.features[-1]):
            self.features[-1][i] = target.unsqueeze(-1)

    def test(self):
        return self.get_dataloader(split="test")

    def eval(self):
        return self.get_dataloader(split="eval")

    def train(self):
        return self.get_dataloader(split="train")

    def get_dataloader(self, split="train"):
        if split == "train":
            sampler = SubsetRandomSampler(self.indices[split])
            shuffle = False  # Sampler handles the shuffling
        else:
            sampler = self.indices[split]  # Use these indices as they are
            shuffle = False
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_batch,
            shuffle=shuffle,  # Sampler makes shuffle unnecessary
        )

    def size(self, split="train"):
        return len(self.indices[split])

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, idx):
        return tuple(feature[idx] for feature in self.features)
