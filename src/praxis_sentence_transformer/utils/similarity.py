# src/utils/similarity.py

import torch
import torch.nn.functional as F

def pytorch_cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Compute cosine similarity between samples in a and b
    
    Parameters:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor
        
    Returns:
        torch.Tensor: Cosine similarity matrix
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
        
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
        
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))