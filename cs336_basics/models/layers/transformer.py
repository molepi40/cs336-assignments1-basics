import torch
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
from einops import rearrange, einsum

from .residue import Residue
from .activation import SwiGLU
from .norm import RMSNorm
from .attention import MultiheadSelfAttention, MultiheadSelfAttentionRoPE

class PreNormTransformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()

        self.attention_norm_layer = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_norm_layer = RMSNorm(d_model, device=device, dtype=dtype)

        self.attention_layer = MultiheadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.swiglu_layer = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        self.attention_residue_layer = Residue()
        self.ffn_residue_layer = Residue()

        self.attention_block = [
            self.attention_norm_layer,
            self.attention_layer,
            self.attention_residue_layer,            
        ]
        self.ffn_block = [
            self.ffn_norm_layer,
            self.swiglu_layer,
            self.ffn_residue_layer,
        ]
    
    def load_weights(
        self,
        attn_norm_weight,
        ffn_norm_weight,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        w1_weight,
        w2_weight,
        w3_weight,        
    ):
        self.attention_norm_layer.load_weights(attn_norm_weight)
        self.ffn_norm_layer.load_weights(ffn_norm_weight)
        
        self.attention_layer.load_weights(
            q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
        )

        self.swiglu_layer.load_weights(
            w1_weight, w2_weight, w3_weight
        )
    
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]):

        self.attention_residue_layer.load_weights(x)
        out1 = x
        for layer in self.attention_block:
            out1 = layer.forward(out1)
        
        self.ffn_residue_layer.load_weights(out1)
        out2 = out1
        for layer in self.ffn_block:
            out2 = layer.forward(out2)
        
        return out2