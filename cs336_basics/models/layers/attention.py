import torch
from torch import Tensor, LongTensor
from jaxtyping import Float, Int
from einops import rearrange, einsum

from .rope import RoPE
from .utils import initialize_linear_weight, softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
):
    d_k = Q.shape[-1]
    q_k = einsum(
        Q, K,
        ' ... queries d_k, ... keys d_k -> ... queries keys'
    )
    q_k = q_k / (d_k ** 0.5)
    if mask is not None:
        q_k = torch.masked_fill(q_k, ~mask, float('-inf'))
    score = softmax(q_k, dim=-1)
    value = einsum(
        score, V,
        ' ... queries keys, ... keys d_v -> ... queries d_v'
    )
    return value


class MultiheadSelfAttention(torch.nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None):
        super().__init__()

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_h = d_model // num_heads

        self.qkv_proj_weight = initialize_linear_weight(d_model, 3 * d_model, device, dtype)
        self.o_proj_weight = initialize_linear_weight(d_model, d_model, device, dtype)
    
    def load_weights(
        self,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
    ):
        # Merge the qkv proj to one proj
        qkv_proj_weight = rearrange(
            [q_proj_weight, k_proj_weight, v_proj_weight],
            ' ... d_in -> (...) d_in'
        )
        # qkv_proj_weight = torch.concatenate([q_proj_weight, k_proj_weight, v_proj_weight], dim=-2)
        assert qkv_proj_weight.shape == (3 * self.d_model, self.d_model)
        self.qkv_proj_weight = torch.nn.Parameter(qkv_proj_weight, requires_grad=True)
        self.o_proj_weight = torch.nn.Parameter(o_proj_weight, requires_grad=True)

    def forward(self, x: Tensor):
        device = x.device
        sequence_length = x.shape[-2]

        # 1. Calculate q k v vectors.
        qkv = einsum(
            x, self.qkv_proj_weight,
            ' ... sequence d_model, d_model_3 d_model -> ... sequence d_model_3'
        )
        # Rearrange q k v vectors in terms of heads.
        qkv_rearranged = rearrange(
            qkv,
            ' ... sequence (num_proj num_heads d_h) -> num_proj ... num_heads sequence d_h',
            num_proj = 3,
            num_heads = self.num_heads
        )
        q, k, v = qkv_rearranged[0], qkv_rearranged[1], qkv_rearranged[2]

        # 2. Create casual mask for sequence.
        mask = torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=0)

        # 3. Calculate self attention and merge vectors from all heads.
        self_attention_value = scaled_dot_product_attention(q, k, v, mask)
        self_attention_value = rearrange(
            self_attention_value,
            ' ... num_heads sequence d_h -> ... sequence (num_heads d_h)',
        )

        # 4. Calculate output vectors.
        output = einsum(
            self_attention_value, self.o_proj_weight,
            ' ... sequence d_v, d_o d_v -> ... sequence d_o'
        )

        return output


class MultiheadSelfAttentionRoPE(MultiheadSelfAttention):
    
    def __init__(
        self, d_model: int, 
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device = None, 
        dtype = None
    ):
        super().__init__(d_model, num_heads, device, dtype)
        # Rope operation is for each head of q and k vectors, so the dimsension is d_h.
        self.rope_layer = RoPE(theta, self.d_h, max_seq_len, device, dtype)

    def forward(self, x: Float[Tensor, ' ... sequence d_in'], token_positions: Int[Tensor, ' ... sequence']=None):
        device = x.device
        sequence_length = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(0, sequence_length)

        # 1. Calculate q k v vectors.
        qkv = einsum(
            x, self.qkv_proj_weight,
            ' ... sequence d_model, d_model_3 d_model -> ... sequence d_model_3'
        )
        # Rearrange q k v vectors in terms of heads.
        qkv_rearranged = rearrange(
            qkv,
            ' ... sequence (num_proj num_heads d_h) -> num_proj ... num_heads sequence d_h',
            num_proj = 3,
            num_heads = self.num_heads
        )
        q, k, v = qkv_rearranged[0], qkv_rearranged[1], qkv_rearranged[2]

        # 2. Calculate q k with rope.
        q = self.rope_layer.forward(q, token_positions)
        k = self.rope_layer.forward(k, token_positions)

        # 3. Create casual mask for sequence.
        mask = torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=0)

        # 4. Calculate self attention and merge vectors from all heads.
        self_attention_value = scaled_dot_product_attention(q, k, v, mask)
        self_attention_value = rearrange(
            self_attention_value,
            ' ... num_heads sequence d_h -> ... sequence (num_heads d_h)',
        )

        # 4. Calculate output vectors.
        output = einsum(
            self_attention_value, self.o_proj_weight,
            ' ... sequence d_v, d_o d_v -> ... sequence d_o'
        )

        return output