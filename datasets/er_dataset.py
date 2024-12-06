import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import torch.utils.data as data
import torch


class ERDataset(data.Dataset):
    """
    """
    def __init__(self, rank, queries_features, database_features, absolute_positives_per_database, predictions, positives_per_query):
        super().__init__()
        self.rank = rank
        self.queries_features = queries_features
        self.database_features = database_features
        self.absolute_positives_per_database = absolute_positives_per_database
        self.predictions = predictions
        self.positives_per_query = positives_per_query
        
    @staticmethod
    def labels_to_ranks(labels: torch.Tensor) -> torch.Tensor:
        """
        Convert binary labels into rank order.
        - Positive labels (1) are ranked first in their original order.
        - Negative labels (0) are ranked after positives in their original order.

        Args:
        - labels (torch.Tensor): Binary labels, shape (N,).

        Returns:
        - ranks (torch.Tensor): Rank order, shape (N,).
        """
        # Indices of positive and negative labels
        positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)
        negative_indices = torch.nonzero(labels == 0, as_tuple=False).squeeze(-1)

        # Initialize ranks tensor
        ranks = torch.zeros_like(labels, dtype=torch.long)

        # Assign ranks
        ranks[positive_indices] = torch.arange(1, len(positive_indices) + 1)
        ranks[negative_indices] = torch.arange(len(positive_indices) + 1, len(labels) + 1)

        return ranks

    def __getitem__(self, index):
        query_feature = self.queries_features[index]
        # random_indices = np.random.choice(20, size=self.rank, replace=False)
        # pres = self.predictions[index][random_indices]
        pres = self.predictions[index][:self.rank]
        neighbors_index = [self.absolute_positives_per_database[pre] for pre in pres]
        neighbors_features_np = np.array([self.database_features[n_index] for n_index in neighbors_index])
        neighbors_features = torch.from_numpy(neighbors_features_np)
        positives = set(self.positives_per_query[index])
        labels = torch.tensor([1 if pre in positives else 0 for pre in pres], dtype=torch.float32)
        ranks = self.labels_to_ranks(labels)
    
        return query_feature, neighbors_features, labels, ranks , index
    
    def __len__(self):
        return len(self.queries_features)
    
    
class ERDataset_test(ERDataset):

    def __getitem__(self, index):
        query_feature = self.queries_features[index]
        pres = self.predictions[index][:self.rank]
        neighbors_index = [self.absolute_positives_per_database[pre] for pre in pres]
        neighbors_features_np = np.array([self.database_features[n_index] for n_index in neighbors_index])
        neighbors_features = torch.from_numpy(neighbors_features_np)
        positives = set(self.positives_per_query[index])
        labels = torch.tensor([1 if pre in positives else 0 for pre in pres], dtype=torch.float32)
        ranks = self.labels_to_ranks(labels)
    
        return query_feature, neighbors_features, labels, ranks , index