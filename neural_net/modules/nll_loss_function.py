import torch

def loss_func(y, alpha, beta): # negative log likelihood (loss function)
    dist = torch.distributions.beta.Beta(alpha, beta)
    neg_log_like = torch.mean(torch.clamp(-dist.log_prob(y), max=10))
    return neg_log_like