import os
import numpy as np
import PIL
from PIL import Image
from scipy import ndimage as ndi
import skimage
from skimage import data
from skimage.morphology import watershed, remove_small_objects
from skimage.filters import sobel, gaussian
from skimage.color import label2rgb
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt

CHANNELS = ['red', 'blue', 'green', 'yellow']


def get_ids(path):
    return list(set(f.split('_')[0] for f in os.listdir(path) if f.endswith("png")))

def merge_rgb(id, input_path, output_path):
    colors = ['red', 'green', 'blue']
    convert_mode = 'L'

    rgb = [PIL.Image.open(os.path.join(input_path, "{}_{}.png".format(id, color))).convert(convert_mode) for color in colors]
    rgb = np.stack(rgb, axis=-1)  # channel last
    img = Image.fromarray(rgb)
    img.save(os.path.join(output_path, "{}.png".format(id)))

def get_mask(img, channel):
    assert channel in CHANNELS

    # (exposure gamma, gaussian filter sigma, object min size, erosion iteration, dilation iteration)
    # arbitrary
    params = {
        "red": (0.5, 3.0, 64, 0, 0),
        "green": (0.5, 2.0, 64, 0, 0),
        "blue": (0.5, 0.0, 16, 0, 0),
        "yellow": (0.5, 2.0, 64, 0, 0)
    }
    gamma, sigma, min_size, erosion, dilation = params.get(channel, (0,0,0,0,0))

    if gamma > 0:
        img = adjust_gamma(img, gamma=gamma)
    if sigma > 0:
        ima = gaussian(img, sigma=sigma) * 255

    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 30] = 1
    markers[img > 150] = 2

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    if min_size > 0:
        segmentation = remove_small_objects(segmentation, min_size)
    if erosion > 0:
        segmentation = ndi.binary_erosion(segmentation, iterations=erosion)
    if dilation > 0:
        segmentation = ndi.binary_dilation(segmentation, iterations=dilation)

    labeled_coins, _ = ndi.label(segmentation)

    num_coin = labeled_coins.max()
    mask = []
    for i in range(1, num_coin+1):  # 0 is black
        m = np.where(labeled_coins==i, 1, 0)
        mask.append(m)
    if not mask:
        empty = np.zeros_like(labeled_coins)
        mask.append(empty)
    mask = np.stack(mask, axis=-1)

    return mask

def get_empty_mask(img):
    empty = np.zeros_like(img)
    mask = np.stack([empty], axis=-1)
    return mask

def mask2img(mask):
    return np.max(mask, axis=-1) * 255

def official_to_mask_png(channel):
    dataset_dir = "data/official/train"
    ids = get_ids(dataset_dir)
    mask_dir = "data/{}_mask".format(channel)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    for id_ in sorted(ids):
        img = skimage.io.imread(os.path.join(dataset_dir, "{}_{}.png".format(id_, channel)))
        mask = get_mask(img, channel)
        print(id_, mask.shape)
        img_mask = mask2img(mask)
        skimage.io.imsave(os.path.join(mask_dir, "{}_{}_mask.png".format(id_, channel)), img_mask)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='valid arguments')
    parser.add_argument('channel', type=str, help='color channel')

    args = parser.parse_args()
    channel = args.channel
    assert channel in CHANNELS

    official_to_mask_png(channel)
