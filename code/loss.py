import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# input = pred value, target = true value

def focal_loss(input, target, gamma=2):
    '''
    https://arxiv.org/abs/1708.02002

    With gamma=0, focal loss is equivalent to binary cross entropy loss.
    https://gombru.github.io/2018/05/23/cross_entropy_loss/
    '''
    # if not (target.size() == input.size()):
    #     raise ValueError("Target size ({}) must be the same as input size ({})"
    #                      .format(target.size(), input.size()))
    #
    # max_val = (-input).clamp(min=0)
    # loss = input - input * target + max_val + \
    #     ((-max_val).exp() + (-input - max_val).exp()).log()
    #
    # invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    # loss = (invprobs * gamma).exp() * loss
    #
    # return loss.sum(dim=1).mean()

    num_class = 28
    size_average = True
    gamma = 2.0

    N = input.size(0)
    C = input.size(1)
    P = F.sigmoid(input)

    class_mask = input.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)

    alpha = Variable(torch.ones(num_class, 1))
    if input.is_cuda and not alpha.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha[ids.data.view(-1)]

    probs = (P*class_mask).sum(1).view(-1,1)

    log_p = probs.log()

    batch_loss = -alpha*(torch.pow((1-probs), gamma))*log_p

    if size_average:
        loss = batch_loss.mean()
    else:
        loss = batch_loss.sum()
    return loss


def f1_loss(input, target, threshold=0.5):
    epsilon = 1e-9
    # true_positive = input * target
    # false_positive = (1 - target) * input
    # false_negative = target * (1 - input)
    # precision = true_positive / (true_positive + false_positive + epsilon)
    # recall = true_positive / (true_positive + false_negative + epsilon)
    # f1 = 2 * precision * recall/(precision + recall + epsilon)
    # return f1.mean()

    # smooth = 0.1
    # epsilon = 1e-4
    # beta = 2
    # beta2 = beta ** 2
    #
    # y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # y_true = K.clip(y_true, epsilon, 1.0 - epsilon)
    #
    # true_and_pred = y_true * y_pred
    # ttp_sum = K.sum(true_and_pred, axis=1)
    # tpred_sum = K.sum(y_pred, axis=1)
    # ttrue_sum = K.sum(y_true, axis=1)
    #
    # tprecision = ttp_sum / tpred_sum
    # trecall = ttp_sum / ttrue_sum
    #
    # tf_score = ((1 + beta2) * tprecision * trecall + smooth) / (beta2 * tprecision + trecall + smooth)
    #
    # return -K.mean(tf_score)

    beta = 1
    input = torch.ge(input.float(), threshold).float()
    target = target.float()
    true_positive = (input * target).sum(dim=1)
    precision = true_positive.div(input.sum(dim=1).add(epsilon))
    recall = true_positive.div(target.sum(dim=1).add(epsilon))
    f1 = torch.mean((precision*recall).div(precision.mul(beta) + recall + epsilon).mul(1 + beta))
    f1 = Variable(f1, requires_grad=True)
    return f1
