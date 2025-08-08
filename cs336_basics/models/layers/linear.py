import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum

from .utils import initialize_linear_weight

class Linear(torch.nn.Module):
    """
    Linear transformation module.
    """
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        device = None, 
        dtype = None,
    ):
        """
        Create weights matrix sized [out_features, in_features].
        """
        super().__init__()

        self.w_weight = initialize_linear_weight(in_features, out_features, device, dtype)  
    
    def load_weights(self, weight: Float[Tensor, " d_out d_in"]):
        self.w_weight = torch.nn.Parameter(weight, requires_grad=True)
        
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Apply the linear transformation to the input
        """
        return  einsum(
            x, self.w_weight,
            "... d_in, d_out d_in -> ... d_out"
        )