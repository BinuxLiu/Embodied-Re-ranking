import torch
import torch.nn as nn
import time

class CrossAttentionReRanker(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads):
        super(CrossAttentionReRanker, self).__init__()
        
        # Cross-Attention for Query-to-Candidate
        self.query_to_candidate_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        
        # Cross-Attention for Candidate-to-Neighbors
        self.candidate_to_neighbors_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        
        # MLP for Fusion and Scoring
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, query, candidates, neighbors):
        """
        Parameters:
        - query: (1, d), the query feature.
        - candidates: (K, d), the candidate features.
        - neighbors: (K, N, d), the nearest neighbors for each candidate.

        Returns:
        - scores: (K,), refined scores for candidates.
        """
        K, N, d = neighbors.size()

        # Step 1: Query-to-Candidate Cross-Attention
        query_expanded = query.expand(K, -1).unsqueeze(0).to(query.device)  # Shape: (1, K, d)
        candidates_input = candidates.unsqueeze(0)                         # Shape: (1, K, d)
        
        query_candidate_context, _ = self.query_to_candidate_attention(
            query_expanded, candidates_input, candidates_input
        )  # Output shape: (1, K, d)

        # Step 2: Candidate-to-Neighbors Cross-Attention
        neighbors_input = neighbors.view(-1, N, d)  # Shape: (K, N, d)
        candidate_context, _ = self.candidate_to_neighbors_attention(
            candidates.unsqueeze(1), neighbors_input, neighbors_input
        )  # Output shape: (K, 1, d)
        candidate_context = candidate_context.squeeze(1)  # Shape: (K, d)

        # Step 3: Fusion of Query Context and Candidate Context
        fusion_features = torch.cat([query_candidate_context.squeeze(0), candidate_context], dim=-1)  # Shape: (K, 2*d)

        # Step 4: Scoring with MLP
        scores = self.fusion_mlp(fusion_features).squeeze(-1)  # Shape: (K,)

        return scores


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example inputs
    feature_dim = 256  # Feature vector size
    hidden_dim = 128   # Hidden dimension for MLP
    num_heads = 8      # Number of attention heads
    K = 10             # Number of candidates
    N = 5              # Number of nearest neighbors per candidate

    # Instantiate the model
    model = CrossAttentionReRanker(feature_dim, hidden_dim, num_heads)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Count parameters
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    # Fake data
    query = torch.randn(1, feature_dim).to(device)             # Query feature: (1, d)
    candidates = torch.randn(K, feature_dim).to(device)        # Candidate features: (K, d)
    neighbors = torch.randn(K, N, feature_dim).to(device)      # Nearest neighbors: (K, N, d)

    # Measure average forward pass time
    torch.cuda.synchronize()  # Ensure GPU timings are accurate
    num_runs = 100
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        scores = model(query, candidates, neighbors)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / num_runs
    print(f"Average forward pass time over {num_runs} runs: {avg_time:.6f} seconds")

    # PyTorch Profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        scores = model(query, candidates, neighbors)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
