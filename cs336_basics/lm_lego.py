import torch
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, state_dict: dict = None):
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


class MultiHeadSelfAttention(torch.nn.Module):
    """ 
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
    """
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.output_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaleDotProductAttention(pdrop=attn_pdrop)

    def forward(self, x):
        B, T, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        mask = torch.triu(torch.ones([T, T]), diagonal=1).bool()
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2) # after transpose, 内存不连续
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        attn_output = self.attn(q, k, v, mask=mask) # attn_output: (B, num_heads, T, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output_proj(attn_output)
    
    def load_state_dict(self, state_dict: dict):
        for i in range(self.num_heads):
            self.q_proj.weight.data[i*self.d_k:(i+1)*self.d_k, :] = state_dict[f"q_heads.{i}.weight"]
            self.k_proj.weight.data[i*self.d_k:(i+1)*self.d_k, :] = state_dict[f"k_heads.{i}.weight"]
            self.v_proj.weight.data[i*self.d_k:(i+1)*self.d_k, :] = state_dict[f"v_heads.{i}.weight"]
        self.output_proj.weight.data = state_dict["output_proj.weight"]
        
class TransformerBlock(torch.nn.Module):
    """
    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, (d_model / num_heads) * num_heads).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
            Shape is (batch_size, sequence_length, d_model).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float, residual_pdrop: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, attn_pdrop=attn_pdrop)
        self.ln1 = RMSNorm(d_model=d_model, eps=1e-5)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model, eps=1e-5)
        self.dropout1 = torch.nn.Dropout(p=residual_pdrop)
        self.dropout2 = torch.nn.Dropout(p=residual_pdrop)
        
    def forward(self, x):
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x
    
if __name__ == "__main__":
    rmsnorm = RMSNorm(d_model=14, eps=1e-8)
    print(rmsnorm.state_dict().keys())
    ffn = FFN(d_model=14, d_ff=14)
    print(ffn.state_dict().keys())
    
    tb = TransformerBlock(d_model=14, num_heads=2, d_ff=14, attn_pdrop=0.1, residual_pdrop=0.1)
    print(tb.state_dict().keys())
