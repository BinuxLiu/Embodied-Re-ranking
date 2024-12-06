import torch
import torch.nn as nn
import time

import torch
import torch.nn as nn
from torch import Tensor

class EmbodiedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.2,
        proj_drop: float = 0.2,
        hidden_dim: int = 64,
    ) -> None:
        """
        Embodied Attention Layer

        Parameters:
        - dim: Feature dimensionality.
        - num_heads: Number of attention heads.
        - qkv_bias: Whether to use bias in Q, K, V projections.
        - proj_bias: Whether to use bias in output projection.
        - attn_drop: Dropout rate for attention weights.
        - proj_drop: Dropout rate for output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = head_dim**-0.5

        # Q, K, V linear projections
        self.q = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, hidden_dim, bias=qkv_bias)

        # Attention dropout and projection
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Scoring layer (MLP to regress similarity score)
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Ranking layer
        self.rank_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, query: Tensor, neighbors: Tensor) -> Tensor:
        """
        Parameters:
        - query: (bs, d), the query feature.
        - neighbors: (bs, num_candidates, num_neighbors, d), the nearest neighbors for each candidate.

        Returns:
        - output: (bs, num_candidates), the similarity score of candidate.
        """
        B, d = query.size()
        num_candidates, num_neighbors, _ = neighbors.size(1), neighbors.size(2), neighbors.size(3)

        # Project query, key, value
        q = self.q(query).unsqueeze(1)  # (bs, 1, d)
        k = self.k(neighbors)          # (bs, num_candidates, num_neighbors, d)
        v = self.v(neighbors)          # (bs, num_candidates, num_neighbors, d)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, 1, -1)  # (bs, num_heads, 1, head_dim)
        k = k.view(B, num_candidates, num_neighbors, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (bs, num_heads, num_candidates, num_neighbors, head_dim)
        v = v.view(B, num_candidates, num_neighbors, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (bs, num_heads, num_candidates, num_neighbors, head_dim)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = torch.einsum("bnhd,bnhkd->bnhk", q, k)  # (bs, num_heads, 1, num_candidates)
        attn = attn.softmax(dim=-1)                    # Normalize over candidates
        attn = self.attn_drop(attn)                    # Apply dropout

        # Aggregate values
        x = torch.einsum("bnhk,bnhkd->bnhd", attn, v)  # (bs, num_heads, 1, head_dim)

        # Concatenate heads and project
        x = x.reshape(B, num_candidates, -1)  # (bs, num_candidates, d)
        scores = self.score_proj(x)
        ranks = self.rank_proj(x)
        
        return scores, ranks



def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    batch_size = 1
    feature_dim = 768
    num_heads = 8
    num_candidates = 10
    num_neighbors = 5

    model = EmbodiedAttention(dim=feature_dim, num_heads=num_heads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    querys = torch.randn(batch_size, feature_dim).to(device)                   # (bs, d)
    neighbors = torch.randn(batch_size, num_candidates, num_neighbors, feature_dim).to(device)   # (bs, k, n, d)

    scores, ranks = model(querys, neighbors)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(scores)
    print(ranks)

