import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self,x):
        return x * torch.rsqrt(x.square().mean(-1,keepdim=True) + self.eps)
                               
    def forward(self,x):
        return self._norm(x.float()).type_as(x) * self.weight
    
