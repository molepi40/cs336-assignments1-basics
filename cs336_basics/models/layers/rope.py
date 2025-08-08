import torch
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
from einops import rearrange, einsum, reduce

class RoPE(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()

        # Theta base with step 2.
        theta_base = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))

        token_positions = torch.arange(0, max_seq_len, 1, device=device, dtype=dtype)

        freqs = einsum(
            token_positions, theta_base,
            ' sequence, rotation -> sequence rotation'
        )
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        self.register_buffer('cos_freqs', cos_freqs)
        self.register_buffer('sin_freqs', sin_freqs)
    
    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"], 
        token_positions: Int[Tensor, " ... sequence_length"]
    ) -> torch.Tensor:
        # 1. Split the sequence into odd and even indices.
        x1, x2 = x[..., 0::2], x[..., 1::2]

        # 2. Indice cos and sin value given by positons of sequence.
        cos_vals = self.cos_freqs[: token_positions.shape[-1]]
        sin_vals = self.sin_freqs[: token_positions.shape[-1]]

        # 3. Calculate odd and even elements of rope vectors.
        rotated_x1 = x1 * cos_vals - x2 * sin_vals # odd
        rotated_x2 = x2 * cos_vals + x1 * sin_vals # even

        # 4. Stack the odd and even elements to form the complete rope vectors.
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated_x = torch.flatten(rotated_x, start_dim=-2, end_dim=-1)
        
        return rotated_x