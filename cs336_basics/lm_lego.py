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
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        x = x - max_val
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)
        return exp_x / sum_exp_x

    
if __name__ == "__main__":
    rmsnorm = RMSNorm(d_model=14, eps=1e-8)
    print(rmsnorm.state_dict().keys())
    ffn = FFN(d_model=14, d_ff=14)
    print(ffn.state_dict().keys())
    
