# Adopted with fastai 1.0 API
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

import os
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from fastai.vision import ImageItemList, ImageDataBunch
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision.models import resnet18
from fastai.metrics import accuracy

num_class = 28
path = "data/rgb_32"
arch = resnet18
size = 32
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
    return (ImageDataBunch.from_csv(path,
                                    csv_labels='train.csv',
                                    folder="train",
                                    test="test",
                                    suffix=".png",
                                    sep=" ",
                                    ds_tfms=tfms,
                                    size=size))

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

print(len(data.train_ds))
print(len(data.test_ds))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def fit():
    learner.fit_one_cycle(5,1e-2)
    learner.save('mini_train_32')

def predict():
    learner.load('mini_train_32')
    preds, y = learner.TTA()
    return preds, y

def save_y(y, th=0.5, fname='output.csv'):
    n_class = len(y[0])
    n_sample = len(y)

    labels = []
    for i in range(n_sample):
        label = ' '.join([str(c) for c in range(n_class) if y[i][c]])
        labels.append(label)

    sample_df = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    sample_list = list(sample_df.Id)

    print(labels)
    print(len(labels))

    pred_dic = dict((key, value) for (key, value)
                in zip(learner.data.test_ds.fnames,labels))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)

if __name__ == '__main__':
    preds, y = predict()
    save_y(y)
