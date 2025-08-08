import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum

from .utils import initialize_linear_weight

class SwiGLU(torch.nn.Module):
    """
    Full forward network layer combining SiLU and GLU. \\
    SiLU: x * sigmoid(x) \\
    GLU: activation(W_1 @ x) * (W_2 @ x) \\
    SwiGLU: W_2 @ (SiLU(W_1 @ x) * (W_3 @ x)) \\
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device = None,
        dtype = None
    ):
        super().__init__()
        
        self.w1_weight = initialize_linear_weight(d_model, d_ff, device, dtype)
        self.w2_weight = initialize_linear_weight(d_ff, d_model, device, dtype)
        self.w3_weight = initialize_linear_weight(d_model, d_ff, device, dtype)
    
    def load_weights(
        self,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
    ):
        self.w1_weight = torch.nn.Parameter(w1_weight, requires_grad=True)
        self.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=True)
        self.w3_weight = torch.nn.Parameter(w3_weight, requires_grad=True)
    
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_ff"]:
        # 1. W_1 @ X
        w1_x = einsum(
            x, self.w1_weight,
            " ... d_model, d_ff d_model -> ... d_ff"
        )

        # 2. X @ W_3
        w3_x = einsum(
            x, self.w3_weight,
            " ... d_model, d_ff d_model -> ... d_ff"
        )

        # 3. SiLU(X @ W_1) * (X @ W_3)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        swiglu_x = silu_w1_x * w3_x
        
        # 4. (SiLU(X @ W_1) * (X @ W_3)) @ W_2
        ff_x = einsum(
            swiglu_x, self.w2_weight,
            " ... d_ff, d_model d_ff -> ... d_model"
        )

        return ff_x
        