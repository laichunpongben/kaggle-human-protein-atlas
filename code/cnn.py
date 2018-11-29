import os
import numpy as np
import pandas as pd
from fastai.vision import ImageItemList
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision.models import resnet18
from fastai.metrics import accuracy

num_class = 28
PATH = "data/rgb"

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
stats = ([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
data = (ImageItemList.from_csv(PATH, 'train.csv', folder="train", suffix='.png')
                     .random_split_by_pct()
                     .label_from_df(sep=' ')
                     .transform(tfms, size=24)
                     .databunch()
                     .normalize(stats))

learner = create_cnn(data, resnet18, metrics=accuracy)
learner.fit_one_cycle(5,1e-2)
learner.save('mini_train')

learner.show_results(figsize=(12,15))

#
# img = learner.data.train_ds[0][0]
# result = learner.predict(img)
# print(result)
