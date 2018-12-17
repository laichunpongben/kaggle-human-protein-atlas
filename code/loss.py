from torch import nn
import torch.nn.functional as F


def focal_loss(input, target, gamma=2):
    '''
    https://arxiv.org/abs/1708.02002

    With gamma=0, focal loss is equivalent to binary cross entropy loss.
    https://gombru.github.io/2018/05/23/cross_entropy_loss/
    '''
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})"
                         .format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
        ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss

    return loss.sum(dim=1).mean()
