import os
import numpy as np
import PIL
from PIL import Image
from scipy import ndimage as ndi
import skimage
from skimage import data
from skimage.morphology import watershed, remove_small_objects
from skimage.filters import sobel
from skimage.color import label2rgb
from skimage.exposure import adjust_gamma
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

def get_mask(img):
    img = adjust_gamma(img, gamma=0.5)  # arbitrary

    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 30] = 1
    markers[img > 150] = 2

    # print(markers)

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    segmentation = remove_small_objects(segmentation, 16)  # arbitrary
    segmentation = ndi.binary_dilation(segmentation, iterations=20)
    labeled_coins, _ = ndi.label(segmentation)
    # print(labeled_coins)

    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
    # ax.set_title('segmentation')
    # ax.axis('off')


    num_coin = labeled_coins.max()
    print(num_coin)
    mask = []
    for i in range(1, num_coin+1):  # 0 is black
        m = np.where(labeled_coins==i, 1, 0)
        mask.append(m)
    if not mask:
        empty = np.zeros_like(labeled_coins)
        mask.append(empty)
    mask = np.stack(mask, axis=-1)


    # plt.tight_layout()
    #
    # plt.show()
    #
    return mask

def mask2img(mask):
    return np.max(mask, axis=-1) * 255


if __name__ == '__main__':
    # input_path = "data/official/test"
    # output_path = "data/rgb/test"
    # ids = get_ids(input_path)
    # print(len(ids))
    # for id_ in ids:
    #     merge_rgb(id_, input_path, output_path)

    dataset_dir = "data/official/train"
    mask_dir = "data/dilated_mask"
    ids = get_ids(dataset_dir)
    for id_ in sorted(ids):
        img = skimage.io.imread(os.path.join(dataset_dir, "{}_blue.png".format(id_)))
        mask = get_mask(img)
        print(id_, mask.shape)
        img_mask = mask2img(mask)
        skimage.io.imsave(os.path.join(mask_dir, "{}_mask.png".format(id_)), img_mask)
