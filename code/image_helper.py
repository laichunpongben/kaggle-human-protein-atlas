import os
import numpy as np
import PIL
from PIL import Image
from scipy import ndimage as ndi
import skimage
from skimage import data, morphology
from skimage.filters import sobel
from skimage.color import label2rgb
import matplotlib.pyplot as plt


def get_ids(path):
    return list(set(f.split('_')[0] for f in os.listdir(path) if f.endswith("png")))

def merge_rgb(id, input_path, output_path):
    colors = ['red', 'green', 'blue']
    convert_mode = 'L'

    rgb = [PIL.Image.open(os.path.join(input_path, "{}_{}.png".format(id, color))).convert(convert_mode) for color in colors]
    rgb = np.stack(rgb, axis=-1)  # channel last
    img = Image.fromarray(rgb)
    img.save(os.path.join(output_path, "{}.png".format(id)))

def red2mask(img):
    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 30] = 1
    markers[img > 150] = 2

    segmentation = morphology.watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    segmentation = morphology.remove_small_objects(segmentation, 9)  # arbitrary
    labeled_coins, _ = ndi.label(segmentation)
    # print(labeled_coins)

    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
    # ax.set_title('segmentation')
    # ax.axis('off')

    num_coin = labeled_coins.max()
    mask = []
    for i in range(1, num_coin+1):  # 0 is black
        m = np.where(labeled_coins==i, 1, 0)
        mask.append(m)
    mask = np.stack(mask, axis=-1)


    # plt.tight_layout()
    #
    # plt.show()
    #
    return mask


if __name__ == '__main__':
    # input_path = "data/official/test"
    # output_path = "data/rgb/test"
    # ids = get_ids(input_path)
    # print(len(ids))
    # for id_ in ids:
    #     merge_rgb(id_, input_path, output_path)

    path = "data/official/train"
    ids = get_ids(path)
    for id_ in sorted(ids):
        img = skimage.io.imread(os.path.join(path, "{}_red.png".format(id_)))
        mask = red2mask(img)
        print(id_, mask.shape)
