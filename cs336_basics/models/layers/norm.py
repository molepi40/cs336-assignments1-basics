import torch
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
from einops import rearrange, einsum, reduce

from .utils import initialize_norm_weight

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.eps = eps

        self.g_weight = initialize_norm_weight(d_model, device, dtype)
    
    def load_weights(self, weight: Float[Tensor, ' d_model']):
        self.g_weight = torch.nn.Parameter(weight, requires_grad=True)
    
    def forward(self, x: Float[Tensor, ' ... d_model']):
        # 1. Convert x to float32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 2. Calculate the square mean of the model dimension.
        mean_square = reduce(x.square(), ' ... d_model -> ...', 'mean')

        # 3. Calculate root mean square of the model dimension.
        rms_norm = x * torch.rsqrt(mean_square + self.eps).unsqueeze(-1) * self.g_weight

        # 4. Convert result to the original data type.
        return rms_norm.to(in_dtype)