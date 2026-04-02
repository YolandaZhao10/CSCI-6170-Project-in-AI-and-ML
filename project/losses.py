import torch
import torch.nn.functional as F
from torch.autograd import grad

def coral_loss(source_features, target_features):
    """
    Computes the CORAL loss (Deep CORAL: Correlation Alignment for Deep Domain Adaptation).
    """
    d = source_features.size(1)
    
    n_s = source_features.size(0)
    tmp_s = torch.ones((1, n_s), device=source_features.device) @ source_features
    cs = (source_features.t() @ source_features - (tmp_s.t() @ tmp_s) / n_s) / (n_s - 1)
    
    n_t = target_features.size(0)
    tmp_t = torch.ones((1, n_t), device=target_features.device) @ target_features
    ct = (target_features.t() @ target_features - (tmp_t.t() @ tmp_t) / n_t) / (n_t - 1)
    
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)
    return loss

def penalty_irm(logits, y):
    """
    Computes the Invariant Risk Minimization (IRMv1) penalty.
    """
    device = logits.device
    scale = torch.tensor(1., device=device, requires_grad=True)
    scaled_logits = logits * scale
    loss = F.cross_entropy(scaled_logits, y)
    grad_scale = grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad_scale ** 2)
