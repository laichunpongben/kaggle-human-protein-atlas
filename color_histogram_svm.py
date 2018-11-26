import os
import itertools
import cv2
import numpy as np
from config import TRAIN_PATH

channels = ["red", "green", "blue", "yellow"]

def get_ids(path):
    return list(set(f.split('_')[0] for f in os.listdir(path)))

def hist_intersection(hist1, hist2):
    minima = np.minimum(hist1, hist2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist2))
    return intersection

def get_intersection(id_):
    imgs = [cv2.imread(os.path.join(TRAIN_PATH, "{}_{}.png".format(id_, channel))) for channel in channels]
    hists = [cv2.calcHist([img], [0], None, [256], [0,256]) for img in imgs]
    return [hist_intersection(*pair) for pair in itertools.combinations(hists, 2)]

if __name__ == '__main__':
    ids = get_ids(TRAIN_PATH)
    for id_ in ids:
        print(id_, get_intersection(id_))
