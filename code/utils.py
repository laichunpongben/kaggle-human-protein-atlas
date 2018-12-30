import math

import cv2
import numpy as np
from fastai.vision.image import *
from fastai.torch_core import to_np

from config import STATS, BASE_NUCLEI_COUNT, BASE_NUCLEI_DENSITY
from .csv_service import get_nuclei_count_density
from .image_service import clipped_zoom

nuclei_count_density = get_nuclei_count_density()

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

    # normalize
    # if "ac1f6b6435d0" in fname:
    #     mean, std = STATS["official"]
    # else:
    #     mean, std = STATS["hpav18"]
    # x = normalize(x, mean, std)

    # zoom
    # nuclei_count, nuclei_density = nuclei_count_density.get(fname, (BASE_NUCLEI_COUNT, BASE_NUCLEI_DENSITY))
    # zoom_scale = get_zoom_scale(nuclei_count, nuclei_density)
    # if not math.isclose(zoom_scale, 1.0):
    #     # if zoom out, padding is handled
    #     x = clipped_zoom(x, zoom_scale)

    return Image(pil2tensor(x, np.float32).float())

def normalize(x, mean, std):
    return (x-mean)/std

def get_zoom_scale(nuclei_count, nuclei_density):
    try:
        return math.sqrt((nuclei_count/BASE_NUCLEI_COUNT)*(BASE_NUCLEI_DENSITY/nuclei_density))
    except ZeroDivisionError:
        return 1.0

def get_stats(data):
    x_tot = np.zeros(4)
    x2_tot = np.zeros(4)

    dls = [data.train_dl, data.valid_dl]
    total_n = sum(len(dl) for dl in dls)
    print(total_n)
    for dl in dls:
        print(dl, len(dl))
        for index, (x,y) in enumerate(dl):
            x = np.moveaxis(to_np(x), 1, -1).reshape(-1,4)  # Shape is bs, channel, imgsize, imgsize. Move channel first to last
            x_tot += x.mean(axis=0)
            x2_tot += (x**2).mean(axis=0)
            print(index, x_tot, x2_tot)

    mean = x_tot/total_n
    std = np.sqrt(x2_tot/total_n - mean**2)
    mean, std = mean.tolist(), std.tolist()
    return mean, std
