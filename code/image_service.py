import os
import io
import re
import time
from pathlib import Path
import argparse

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
import cv2
import matplotlib.pyplot as plt
import lzma
import libarchive.public

CHANNELS = ['red', 'blue', 'green', 'yellow']
PATH = "/home/ben/Downloads"


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

def resize_png(dataset_dir, output_dir, size, id_=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if id_:
        imgs = [id_]
    else:
        imgs = list(set(f.split('.png')[0] for f in os.listdir(dataset_dir) if f.endswith("png")))

    for id_ in sorted(imgs):
        try:
            img = skimage.io.imread(os.path.join(dataset_dir, "{}.png".format(id_)))
            img = skimage.transform.resize(img, (size, size))
            skimage.io.imsave(os.path.join(output_dir, "{}.png".format(id_)), img)
        except Exception as e:
            print(e)

def test_imread(dataset_dir, size=0):
    channel = "blue"
    funcs = {
        "OpenCV": (cv2.imread, (cv2.IMREAD_GRAYSCALE, ), {}),
        "Scikit image": (skimage.io.imread, (), {"as_gray": True})
    }

    ids = get_ids(dataset_dir)

    for k, v in funcs.items():
        func, args, kwargs = v
        start = time.time()
        for index, id_ in enumerate(sorted(ids)):
            if 0 < size <= index:
                break
            img = func(os.path.join(dataset_dir, "{}_{}.png".format(id_, channel)), *args, **kwargs)
            mask = get_mask(img, channel)

        end = time.time()
        lapsed = end - start
        print("{}: {:15f}".format(k, lapsed))

def test_imread_collection(dataset_dir, size=0):
    channel = "blue"
    start = time.time()
    collection = skimage.io.imread_collection(os.path.join(dataset_dir, "*_{}.png".format(channel)))
    for index, img in enumerate(collection):
        if 0 < size <= index:
            break
        mask = get_mask(img, channel)
    end = time.time()
    lapsed = end - start
    print("Scikit image collection: {:15f}".format(lapsed))

def test():
    test_size = 1000
    dataset_dir = "data/official/train"
    test_imread(dataset_dir, size=test_size)
    test_imread_collection(dataset_dir, size=test_size)

def check_colors(dataset_dir):
    fs = sorted(os.listdir(dataset_dir))
    ids = list(set([re.sub("(_red|_green|_blue|_yellow)", "", f.split(".png")[0]) for f in os.listdir(dataset_dir)]))
    missing_ids = []
    for id_ in ids:
        count = sum(1 for f in fs if f.startswith(id_))
        if count < 4:
            print(id_)
            missing_ids.append(id_)
    return missing_ids

def convert_grayscale(f, in_, out):
    img = Image.open(os.path.join(in_, f)).convert('L')
    img.save(os.path.join(out, f))

def flann_match_images(dataset_dir0, dataset_dir1, channel=None):
    if channel:
        fs0 = [f for f in os.listdir(dataset_dir0) if f.endswith("_{}.png".format(channel))]
        fs1 = [f for f in os.listdir(dataset_dir1) if f.endswith("_{}.png".format(channel))]
    else:
        fs0 = [f for f in os.listdir(dataset_dir0) if f.endswith(".png")]
        fs1 = [f for f in os.listdir(dataset_dir1) if f.endswith(".png")]
    imgs1 = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs1]

    sift = cv2.xfeatures2d.SIFT_create()
    details0 = []
    for f in fs0:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        details0.append((f, kp, des))

    details1 = []
    for f in fs1:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        details1.append((f, kp, des))

    details0.sort(key=lambda x: x[2])
    details1.sort(key=lambda x: x[2])

    print(details0)
    print(details1)
    print('Loaded all sift details')

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    print('Flann')

def clipped_zoom(img, zoom_factor):
    """
    https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions

    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def tif2png(sz):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--ds", help="dataset", type=str, choices=["train", "test"], required=True)
    args = parser.parse_args()
    ds = args.ds
    filename = ds + "_full_size"

    fnames = os.listdir(Path(PATH)/'{}'.format(ds))
    pattern = re.compile("NAME=\[(.+?)\] SIZE")

    zip_path = Path(PATH)/f'{filename}.7z'
    with libarchive.public.file_reader(zip_path.__str__()) as f0:
        # x = sum(1 for _ in f0)
        # print(x)
        for entry in f0:
            tif_name = re.search(pattern, str(entry)).group(1)
            print(tif_name)
            if ".tif" not in tif_name:
                continue

            if tif_name in [
                # "a14399ee-bad4-11e8-b2b8-ac1f6b6435d0_yellow.tif",
                # "d67b7b9a-bac5-11e8-b2b7-ac1f6b6435d0_yellow.tif",
                # "fcb8e0b6-bad6-11e8-b2b9-ac1f6b6435d0_green.tif"
                "8ba4bc58-bbb5-11e8-b2ba-ac1f6b6435d0_yellow.tif",
                "a9125fa6-bbbb-11e8-b2ba-ac1f6b6435d0_yellow.tif"
            ]:
                continue

            name = tif_name[:-4] + ".png"
            if name in fnames:
                continue

            tif_path = Path(PATH)/'{}'.format(ds)/f'{tif_name}'
            png_path = Path(PATH)/'{}'.format(ds)/f'{name}'

            # try:
            with open(tif_path, 'wb') as f1:
                blocks = []
                for block in entry.get_blocks():
                    f1.write(block)
            img = Image.open(tif_path)
            img = img.resize((sz, sz))
            img.save(png_path, format="png")
            print("Saved {}".format(name))
            # except Exception as e:
            #     print(e)
            # finally:
            Path.unlink(tif_path)
            # break
            # with
        # print(content)
        # for name in compressed:
        #     print(name)
        #     img = Image.open(io.BytesIO(archive.read(name)))
        #     output = io.BytesIO()
        #     img.resize((sz,sz)).save(output, format='png')
        #     archive_out.writestr(name, output.getvalue())

if __name__ == '__main__':
    # missing_ids = check_colors("data/hpav18")
    sz = 1024
    tif2png(sz)
    # dataset_dir = "/home/ben/github/atlas/data/hpav18/train"
    # output_dir = "/home/ben/Downloads"
    # size = sz
    # id_ = "a9125fa6-bbbb-11e8-b2ba-ac1f6b6435d0_yellow"
    # resize_png(dataset_dir, output_dir, size, id_=id_)
