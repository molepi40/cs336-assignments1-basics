import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import reduce

def initialize_linear_weight(in_features: int, out_features: int, device, dtype):
    delta = 2.0 / (in_features + out_features)
    weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
    weight = torch.nn.init.trunc_normal_(weight, mean=0, std=delta, a=-3*delta, b=3*delta)
    weight = torch.nn.Parameter(weight, requires_grad=True)

    return weight

def initialize_embedding_weight(num_embeddings: int, embedding_dim: int, device, dtype):
    weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
    weight = torch.nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
    weight = torch.nn.Parameter(weight, requires_grad=True)

    return weight

def initialize_norm_weight(model_dim: int, device, dtype):
    weight = torch.ones((model_dim), device=device, dtype=dtype)
    weight = torch.nn.Parameter(weight, requires_grad=True)

    return weight

def softmax(x: Tensor, dim: int): 
    safe_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    exp_safe_x = torch.exp(safe_x)
    safe_softmax = exp_safe_x / torch.sum(exp_safe_x, dim=dim, keepdim=True)
    
    return safe_softmax

def cross_entropy(logits: Float[Tensor, ' ... vocab_size'], target: Int[Tensor, ' ...']):
    # 1. Get safe logist by subtract the max element.
    safe_logist = logits - torch.max(logits, dim=-1, keepdim=True)[0]

    # 2. Calculate sum of exp(safe_logists).
    exp_safe_logist = torch.exp(safe_logist)
    exp_safe_logist_sum = torch.sum(exp_safe_logist, dim=-1)

    # 3. Gather the target in logists.
    target_safe_logist = torch.gather(safe_logist, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # 4. Calculate the cross entropy loss
    loss = torch.log(exp_safe_logist_sum) - target_safe_logist

    # 5. Get the mean loss across batch. 
    res = reduce(
        loss,
        ' batch -> ',
        'mean'
    )

    return res