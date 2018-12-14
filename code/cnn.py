# Adopted with fastai 1.0 API
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from fastai.vision import ImageItemList, ImageDataBunch
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision.models import resnet18, resnet34
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
    '''
    Prevent the following RuntimeError:
    Expected object of scalar type Long but got scalar type Float for argument #2 'other'
    '''
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
    learner.crit = nn.BCEWithLogitsLoss()
    # learner.crit = FocalLoss()
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
    learner.save('bce_32')

def tta():
    learner.load('bce_32')
    preds, y = learner.TTA()
    return preds, y

def predict(fname='output.csv'):
    th = [
        0.0104, #0: 0.41468202883
        0.191, #1: 0.04035787847
        0.0091, #2: 0.11653578784
        0.0015, #3: 0.05023815653
        0.545, #4: 0.05979660144
        0.12, #5: 0.08087667353
        0.088, #6: 0.03244078269
        0.09, #7: 0.09082131822
        0.12, #8: 0.00170571575
        0.16, #9: 0.00144824922
        0.07, #10: 0.00090113285
        0.7, #11: 0.03517636457
        0.3, #12: 0.02214212152
        0.09, #13: 0.01728244078
        0.148, #14: 0.03430741503
        0.00025, #15: 0.00067584963
        0.102, #16: 0.01705715756
        0.269, #17: 0.00675849639
        0.026, #18: 0.02902935118
        0.074, #19: 0.04769567456
        0.0107, #20: 0.00553553038
        0.15, #21: 0.12155638517
        0.08, #22: 0.02581101956
        0.019, #23: 0.09542353244
        0.0031, #24: 0.0103630278
        0.051, #25: 0.26480432543
        0.056, #26: 0.0105561277
        0.015 #27: 0.00035401647
    ]

    n_y = len(data.test_ds)

    learner.load('bce_32')

    labels = []
    cls_count = defaultdict(int)
    max_ = 11702
    for i in range(n_y):
        if i >= max_:
            break

        img = data.test_ds[i][0]
        pred = learner.predict(img)
        classes = []
        for c in range(num_class):
            if pred[0][c]>th[c]:
                classes.append(str(c))
                cls_count[c] += 1

        label = ' '.join(classes)
        if not label:
            label = '0'
        labels.append(label)

    print(cls_count)
    for k,v in sorted(cls_count.items()):
        print(k, v/max_)

    sample_df = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    sample_list = list(sample_df.Id)

    df = pd.DataFrame({'Id':sample_list,'Predicted':labels})
    df.to_csv(fname, header=True, index=False)

if __name__ == '__main__':
    # fit()
    predict()
