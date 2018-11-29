import os
import numpy as np
import pandas as pd
from fastai.vision import ImageDataBunch
from fastai.vision.transform import rand_resize_crop
from fastai.vision.learner import create_cnn
from fastai.vision.models import resnet18
from fastai.metrics import accuracy
from config import TRAIN_PATH, TEST_PATH

num_class = 28
PATH = "data/rgb"

def to_category(list_, max_=num_class):
    category = [0] * max_
    for x in list_:
        category[int(x)] = 1
    return category


train = pd.read_csv(os.path.join(TRAIN_PATH, "../train.csv"))
# get y values
train['Target'] = train['Target'].str.split(' ')
train['Categories'] = list(train['Target'].map(to_category))
y_columns = list(range(num_class))
train[y_columns] = pd.DataFrame(train.Categories.values.tolist(), index=train.index)
train = train.drop(['Target', 'Categories'], axis=1)

# tfms = rand_resize_crop(24)
# print(tfms)
data = ImageDataBunch.from_df("data/rgb/train",
                              train,
                              suffix=".png",
                              fn_col="Id",
                              label_col=list(range(num_class)),
                              size=24,
                              bs=16)  # df_tfms=tfms,


learner = create_cnn(data, resnet18, metrics=accuracy)
learner.fit(1)
learner.save('one_epoch')

img = learner.data.train_ds[0][0]
result = learner.predict(img)
print(result)
