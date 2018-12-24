import math
from pathlib import Path
import time

import cv2
import numpy as np
from fastai.vision.image import *

from config import STATS, BASE_NUCLEI_COUNT, BASE_NUCLEI_DENSITY
from .csv_service import get_nuclei_count_density
from .image_service import clipped_zoom

nuclei_count_density = get_nuclei_count_density()
debug_counter = 0
debug_time = time.time()
open_time = 0.0

# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    global debug_counter, debug_time, open_time
    start = time.time()
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]

    x = np.stack(img, axis=-1)
    # np.savez_compressed(Path("data/npz/train")/f'{fname}.npz', x)

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
    img = Image(pil2tensor(x, np.float32).float())
    now = time.time()
    open_time += now - start
    if debug_counter % 100 == 0:
        print(now - debug_time, open_time, open_time/(now - debug_time))
        debug_time = now
        open_time = 0.0
    debug_counter += 1

    return img

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
    for x,y in iter(data.train_dl):
        x = np.moveaxis(to_np(x), 1, -1).reshape(-1,4)  # Shape is bs, channel, imgsize, imgsize. Move channel first to last
        x_tot += x.mean(axis=0)
        x2_tot += (x**2).mean(axis=0)

    mean = x_tot/len(data.train_dl)
    std = np.sqrt(x2_tot/len(data.train_dl) - mean**2)
    mean, std = mean.tolist(), std.tolist()
    return mean, std
