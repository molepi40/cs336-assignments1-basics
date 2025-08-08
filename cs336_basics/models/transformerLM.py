import torch
from torch import Tensor
from jaxtyping import Float, Int
from .layers.embedding import Embedding
from .layers.transformer import PreNormTransformer
from .layers.norm import RMSNorm
from .layers.linear import Linear
from .layers.utils import softmax

class TransformerLM(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int, context_length: int, num_layers: int, num_heads:int, 
        d_model: int, d_ff: int, rope_theta: float,
        device=None, dtype=None
    ):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transform_layers = [
            PreNormTransformer(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) \
            for i in range(num_layers)
        ]
        self.norm_layer = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear_layer = Linear(d_model, vocab_size, device=device, dtype=dtype)

        self.layers = [
            self.embedding_layer,
            self.transform_layers,
            self.norm_layer,
            self.linear_layer
        ]
    
    def load_weights_from_dict(self, weights_dict: dict[str, Tensor]):
        self.embedding_layer.load_weights(weights_dict['token_embeddings.weight'])

        for layer_idx, transform_layer in enumerate(self.transform_layers):
            transform_layer.load_weights(
                weights_dict[f'layers.{layer_idx}.ln1.weight'],
                weights_dict[f'layers.{layer_idx}.ln2.weight'],
                weights_dict[f'layers.{layer_idx}.attn.q_proj.weight'],
                weights_dict[f'layers.{layer_idx}.attn.k_proj.weight'],
                weights_dict[f'layers.{layer_idx}.attn.v_proj.weight'],
                weights_dict[f'layers.{layer_idx}.attn.output_proj.weight'],
                weights_dict[f'layers.{layer_idx}.ffn.w1.weight'],
                weights_dict[f'layers.{layer_idx}.ffn.w2.weight'],
                weights_dict[f'layers.{layer_idx}.ffn.w3.weight'],
            )
        
        self.norm_layer.load_weights(weights_dict['ln_final.weight'])
        self.linear_layer.load_weights(weights_dict['lm_head.weight'])
    
    def forward(self, x: Float[Tensor, ' ... sequence d_in']):
        out = x
        for layer in self.layers:
            if isinstance(layer, list):
                for sub_layer in layer:
                    out = sub_layer.forward(out)
            else:
                out = layer.forward(out)
        
        return out