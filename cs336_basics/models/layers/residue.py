import torch
from torch import Tensor
from jaxtyping import Float

class Residue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = None
    
    def load_weights(self, x: Float[Tensor, '... sequence d_x']):
        self.x = x
    
    def forward(self, x: Float[Tensor, '... sequence d_x']):
        return self.x + x