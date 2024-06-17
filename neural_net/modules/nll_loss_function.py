"""
============================================================================================
                                 nll_loss_function.py

Python file containing loss function for neural networks. The loss function is a negative
log likelihood implemetation of the beta distribution.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch

def loss_func(y, alpha, beta): # negative log likelihood (loss function)
    """loss_func
    Loss function for neural networks.

    Parameters:
    y: label value (value network is trying to predict)
    alpha: alpha parameter of predicted beta distribution
    beta: beta parameter of predicted beta distribution

    Returns:
    neg_log_like: negative log likelihood of prediction
    """
    dist = torch.distributions.beta.Beta(alpha, beta)
    neg_log_like = torch.mean(torch.clamp(-dist.log_prob(y), max=10))
    return neg_log_like