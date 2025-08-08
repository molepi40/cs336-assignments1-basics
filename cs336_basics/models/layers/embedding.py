import torch
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
from einops import rearrange, einsum

from .utils import initialize_embedding_weight

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        
        self.embedding_weight = initialize_embedding_weight(num_embeddings, embedding_dim, device, dtype)
    
    def load_weights(self, weight: Float[Tensor, 'vocab_size d_model']):
        self.embedding_weight = torch.nn.Parameter(weight, requires_grad=True)
    
    def forward(self, x: Float[LongTensor, ' ...']) -> Float[Tensor, ' ... d_model']:

        return self.embedding_weight[x]