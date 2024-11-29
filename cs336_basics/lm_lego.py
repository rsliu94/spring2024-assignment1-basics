import torch

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
    
if __name__ == "__main__":
    rmsnorm = RMSNorm(d_model=14, eps=1e-8)
    print(rmsnorm.state_dict())
