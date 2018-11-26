import os
import itertools
import cv2
import numpy as np
import pandas as pd
from config import TRAIN_PATH

channels = ["red", "green", "blue", "yellow"]
num_class = 28

def get_ids(path):
    return list(set(f.split('_')[0] for f in os.listdir(path)))

def hist_intersection(hist1, hist2):
    minima = np.minimum(hist1, hist2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist2))
    return intersection

def get_intersections(id_):
    imgs = [cv2.imread(os.path.join(TRAIN_PATH, "{}_{}.png".format(id_, channel))) for channel in channels]
    hists = [cv2.calcHist([img], [0], None, [256], [0,256]) for img in imgs]
    return [hist_intersection(*pair) for pair in itertools.combinations(hists, 2)]

def get_train():
    train = pd.read_csv(os.path.join(TRAIN_PATH, "../train.csv"), index_col='Id')

    # get x values
    x_columns = ["RG", "RB", "RY", "GB", "GY", "BY"]
    train['Intersections'] = train.index.map(get_intersections)
    train[x_columns] = pd.DataFrame(train.Intersections.values.tolist(), index=train.index)

    # get y values
    train['Target'] = train['Target'].str.split(' ')
    train['Categories'] = list(train['Target'].map(to_category))
    y_columns = list(range(num_class))
    train[y_columns] = pd.DataFrame(train.Categories.values.tolist(), index=train.index)

    train = train.drop(['Target', 'Intersections', 'Categories'], axis=1)
    return train

def to_category(list_, max_=num_class):
    category = [0] * max_
    for x in list_:
        category[int(x)] = 1
    return category

if __name__ == '__main__':
    train = get_train()
    print(train)
