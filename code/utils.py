import cv2
import numpy as np

from fastai.vision.image import *

from config import STATS

# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]

    x = np.stack(img, axis=-1)

    if "ac1f6b6435d0" in fname:
        mean, std = STATS["official"]
    else:
        mean, std = STATS["hpav18"]
    x = normalize(x, mean, std)

    return Image(pil2tensor(x, np.float32).float())

def normalize(x, mean, std):
    return (x-mean)/std


def get_stats(data):
    x_tot = np.zeros(4)
    x2_tot = np.zeros(4)
    for x,y in iter(data.train_dl):
        x = np.moveaxis(to_np(x), 1, -1).reshape(-1,4)  # Shape is bs, channel, imgsize, imgsize. Move channel first to last
        x_tot += x.mean(axis=0)
        x2_tot += (x**2).mean(axis=0)

    mean = x_tot/len(data.train_dl)
    std = np.sqrt(x2_tot/len(data.train_dl) - mean**2)
    mean, std = mean.tolist(), std.tolist()
    return mean, std
