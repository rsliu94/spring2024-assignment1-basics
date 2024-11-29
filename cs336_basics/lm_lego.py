import torch
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8, state_dict: dict = None):
        super().__init__()
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        assert x.shape[-1] == self.d_model
        denominator = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1, keepdim=True) / self.d_model + self.eps)
        return self.weight * x / denominator
    
class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    
class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()
    
    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))
    
class Softmax(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]
        x = x - max_val
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp_x
    
class ScaleDotProductAttention(torch.nn.Module):
    """
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
    """
    def __init__(self, pdrop: float = None):
        super().__init__()
        self.pdrop = pdrop
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -torch.inf)
        softmax = Softmax(dim=-1)
        attention_probs = softmax(scores)
        if self.pdrop is not None:
            attention_probs = torch.nn.functional.dropout(attention_probs, p=self.pdrop)
        return attention_probs @ V

    
if __name__ == "__main__":
    rmsnorm = RMSNorm(d_model=14, eps=1e-8)
    print(rmsnorm.state_dict().keys())
    ffn = FFN(d_model=14, d_ff=14)
    print(ffn.state_dict().keys())
    
