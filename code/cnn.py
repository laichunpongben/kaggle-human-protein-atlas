# Adopted with fastai 1.0 API
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

import os
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from fastai.vision import ImageItemList
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision.models import resnet18
from fastai.metrics import accuracy

num_class = 28
path = "data/rgb"
arch = resnet18
size = 128
stats = ([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def get_tfms():
    return get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

def get_data(tfms):
    return (ImageItemList.from_csv(path, 'train.csv', folder="train", suffix='.png')
                         .random_split_by_pct()
                         .label_from_df(sep=' ')
                         .transform(tfms, size=size)
                         .databunch()
                         .normalize(stats))

def get_learner(data):
    learner = create_cnn(data, arch, ps=0.5)
    learner.opt_fn = Adam
    learner.clip = 1.0 #gradient clipping
    learner.crit = FocalLoss()
    learner.metrics = [acc]
    return learner

tfms = get_tfms()
data = get_data(tfms)
learner = get_learner(data)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def fit():
    learner.fit_one_cycle(5,1e-2)
    learner.save('mini_train_128')
    print('model fitted and saved')

def predict():
    if os.path.isfile('mini_train_128') :
        learner.load('mini_train_128')
        print('loaded model')
    else:
        # ValueError: padding_mode needs to be 'zeros' or 'border', but got reflection (fastai 1.0.31)
        # fastai/vision/image.py line 92 and 503: padding_mode changed from 'reflection' to 'zeros' to avoid error
        fit()
    preds, y = learner.TTA()
    print(preds)
    print(y)
    preds = np.stack(preds, axis=-1)
    preds = sigmoid(preds)
    print(preds)
    pred = preds.max(axis=-1) #max works better for F1 macro score
    return pred

def save_pred(pred, th=0.5, fname='output.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)

    sample_df = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value)
                in zip(learner.data.test_ds.fnames,pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)

if __name__ == '__main__':
    pred = predict()
    print(pred)
    save_pred(pred)