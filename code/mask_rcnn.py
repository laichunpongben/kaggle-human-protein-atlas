# Adopted to Human Protein Atlas

"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from atlas.code.image_service import get_mask

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    "af023944-bbc6-11e8-b2bc-ac1f6b6435d0",
    "af0497e2-bbb8-11e8-b2ba-ac1f6b6435d0",
    "af04b13c-bbbd-11e8-b2ba-ac1f6b6435d0",
    "af04c52c-bbc8-11e8-b2bc-ac1f6b6435d0",
    "af059696-bbaa-11e8-b2ba-ac1f6b6435d0",
    "af05c5a8-bbb9-11e8-b2ba-ac1f6b6435d0",
    "af08c27c-bb9a-11e8-b2b9-ac1f6b6435d0",
    "af09cea6-bb9f-11e8-b2b9-ac1f6b6435d0",
    "af0a2d3e-bb99-11e8-b2b9-ac1f6b6435d0",
    "af0aea6a-bbb4-11e8-b2ba-ac1f6b6435d0",
    "af0b0af6-bba8-11e8-b2ba-ac1f6b6435d0",
    "af15fab8-bbbe-11e8-b2ba-ac1f6b6435d0",
    "af18c26c-bbc2-11e8-b2bc-ac1f6b6435d0",
    "af19bb12-bbaf-11e8-b2ba-ac1f6b6435d0",
    "af1cbcc4-bba8-11e8-b2ba-ac1f6b6435d0",
    "af225d18-bbb3-11e8-b2ba-ac1f6b6435d0",
    "af26bdf4-bbae-11e8-b2ba-ac1f6b6435d0",
    "af2e8574-bbc8-11e8-b2bc-ac1f6b6435d0",
    "af34bdbc-bbbf-11e8-b2bb-ac1f6b6435d0",
    "af37da38-bba6-11e8-b2ba-ac1f6b6435d0",
    "af38cae4-bbb8-11e8-b2ba-ac1f6b6435d0",
    "af3b3de6-bba0-11e8-b2b9-ac1f6b6435d0",
    "af3d7206-bbb8-11e8-b2ba-ac1f6b6435d0",
    "af3f6482-bbc5-11e8-b2bc-ac1f6b6435d0",
    "af44125e-bbbe-11e8-b2ba-ac1f6b6435d0",
    "af4644f6-bbc9-11e8-b2bc-ac1f6b6435d0",
    "af46a966-bba4-11e8-b2ba-ac1f6b6435d0",
    "af46acd4-bb9c-11e8-b2b9-ac1f6b6435d0",
    "af47025c-bbaa-11e8-b2ba-ac1f6b6435d0",
    "af477ebe-bbb2-11e8-b2ba-ac1f6b6435d0",
    "af48825a-bbc6-11e8-b2bc-ac1f6b6435d0",
    "af49878c-bba7-11e8-b2ba-ac1f6b6435d0",
    "af4ab20c-bbbf-11e8-b2bb-ac1f6b6435d0",
    "af4acb7e-bbb6-11e8-b2ba-ac1f6b6435d0",
    "af4b8a12-bbb3-11e8-b2ba-ac1f6b6435d0",
    "af4ea060-bbc9-11e8-b2bc-ac1f6b6435d0",
    "af4fc406-bba9-11e8-b2ba-ac1f6b6435d0",
    "af516262-bbbc-11e8-b2ba-ac1f6b6435d0",
    "af545a64-bbc9-11e8-b2bc-ac1f6b6435d0",
    "af5463f6-bbab-11e8-b2ba-ac1f6b6435d0",
    "af577c36-bba5-11e8-b2ba-ac1f6b6435d0",
    "af5ce556-bbb7-11e8-b2ba-ac1f6b6435d0",
    "af611c58-bbca-11e8-b2bc-ac1f6b6435d0",
    "af623c04-bbbc-11e8-b2ba-ac1f6b6435d0",
    "af6308b8-bba9-11e8-b2ba-ac1f6b6435d0",
    "af69d0ae-bba3-11e8-b2b9-ac1f6b6435d0",
    "af6b4406-bbae-11e8-b2ba-ac1f6b6435d0",
    "af6c9f66-bbab-11e8-b2ba-ac1f6b6435d0",
    "af6d2402-bbad-11e8-b2ba-ac1f6b6435d0",
    "af6eb854-bbb6-11e8-b2ba-ac1f6b6435d0",
    "af732b88-bbb0-11e8-b2ba-ac1f6b6435d0",
    "af73629a-bbb2-11e8-b2ba-ac1f6b6435d0",
    "af75165e-bbc5-11e8-b2bc-ac1f6b6435d0",
    "af7636d8-bbac-11e8-b2ba-ac1f6b6435d0",
    "af7648f0-bbbe-11e8-b2ba-ac1f6b6435d0",
    "af796d8a-bbb6-11e8-b2ba-ac1f6b6435d0",
    "af7e3ef2-bbc3-11e8-b2bc-ac1f6b6435d0",
    "af7eeb3c-bbc7-11e8-b2bc-ac1f6b6435d0",
    "af80ae16-bbb5-11e8-b2ba-ac1f6b6435d0",
    "af87c876-bbb6-11e8-b2ba-ac1f6b6435d0",
    "af881852-bba3-11e8-b2b9-ac1f6b6435d0",
    "af8aa366-bbb6-11e8-b2ba-ac1f6b6435d0",
    "af8b187c-bbc1-11e8-b2bb-ac1f6b6435d0",
    "af8bac12-bbab-11e8-b2ba-ac1f6b6435d0",
    "af8bd114-bbca-11e8-b2bc-ac1f6b6435d0",
    "af938298-bbab-11e8-b2ba-ac1f6b6435d0",
    "af93bf5c-bb9b-11e8-b2b9-ac1f6b6435d0",
    "af9481da-bbaf-11e8-b2ba-ac1f6b6435d0",
    "af955a38-bbaf-11e8-b2ba-ac1f6b6435d0",
    "af96c00a-bba3-11e8-b2b9-ac1f6b6435d0",
    "af9b07be-bbb7-11e8-b2ba-ac1f6b6435d0",
    "af9b6d76-bb99-11e8-b2b9-ac1f6b6435d0",
    "af9b8440-bb9f-11e8-b2b9-ac1f6b6435d0",
    "af9edc70-bbb9-11e8-b2ba-ac1f6b6435d0",
    "afa0f196-bba4-11e8-b2ba-ac1f6b6435d0",
    "afa1ca18-bb9e-11e8-b2b9-ac1f6b6435d0",
    "afa258c2-bbbf-11e8-b2bb-ac1f6b6435d0",
    "afa94674-bbb4-11e8-b2ba-ac1f6b6435d0",
    "afab025c-bbbe-11e8-b2ba-ac1f6b6435d0",
    "afab569c-bbb6-11e8-b2ba-ac1f6b6435d0",
    "afac1158-bbc6-11e8-b2bc-ac1f6b6435d0",
    "afb14368-bba7-11e8-b2ba-ac1f6b6435d0",
    "afb206ea-bbca-11e8-b2bc-ac1f6b6435d0",
    "afb33eaa-bbc7-11e8-b2bc-ac1f6b6435d0",
    "afb651ca-bbc8-11e8-b2bc-ac1f6b6435d0",
    "afb9000c-bbc1-11e8-b2bb-ac1f6b6435d0",
    "afbb19b2-bbbe-11e8-b2ba-ac1f6b6435d0",
    "afbbbaf2-bbc8-11e8-b2bc-ac1f6b6435d0",
    "afbfa3f4-bbac-11e8-b2ba-ac1f6b6435d0",
    "afc29062-bbae-11e8-b2ba-ac1f6b6435d0",
    "afc313c8-bbbf-11e8-b2bb-ac1f6b6435d0",
    "afc5a964-bbb3-11e8-b2ba-ac1f6b6435d0",
    "afc70c00-bbb3-11e8-b2ba-ac1f6b6435d0",
    "afc756f6-bbc2-11e8-b2bc-ac1f6b6435d0",
    "afc82a04-bbb3-11e8-b2ba-ac1f6b6435d0",
    "afccc80e-bbca-11e8-b2bc-ac1f6b6435d0",
    "afce3952-bba0-11e8-b2b9-ac1f6b6435d0",
    "afce60b8-bbc2-11e8-b2bc-ac1f6b6435d0",
    "afceee7c-bbc5-11e8-b2bc-ac1f6b6435d0",
    "afd2c640-bbb2-11e8-b2ba-ac1f6b6435d0",
    "afd79f18-bbc0-11e8-b2bb-ac1f6b6435d0",
    "afd86c92-bbaf-11e8-b2ba-ac1f6b6435d0",
    "afd8964e-bba6-11e8-b2ba-ac1f6b6435d0",
    "afd97e9a-bbad-11e8-b2ba-ac1f6b6435d0",
    "afda66fa-bba0-11e8-b2b9-ac1f6b6435d0",
    "afdd0490-bb9c-11e8-b2b9-ac1f6b6435d0",
    "afe61d42-bbb4-11e8-b2ba-ac1f6b6435d0",
    "afe69cd6-bbc3-11e8-b2bc-ac1f6b6435d0",
    "afe6f0dc-bba5-11e8-b2ba-ac1f6b6435d0",
    "afe79c2e-bbb1-11e8-b2ba-ac1f6b6435d0",
    "afe7c11e-bbab-11e8-b2ba-ac1f6b6435d0",
    "afede996-bbaa-11e8-b2ba-ac1f6b6435d0",
    "afef49a2-bbc4-11e8-b2bc-ac1f6b6435d0",
    "aff6b050-bbb2-11e8-b2ba-ac1f6b6435d0",
    "aff7af2c-bb9f-11e8-b2b9-ac1f6b6435d0",
    "affa11e2-bba2-11e8-b2b9-ac1f6b6435d0",
    "affbfb76-bba0-11e8-b2b9-ac1f6b6435d0",
    "affd4f82-bb9e-11e8-b2b9-ac1f6b6435d0",
    "affda574-bbc9-11e8-b2bc-ac1f6b6435d0",
    "affdc134-bba2-11e8-b2b9-ac1f6b6435d0",
    "affe2a62-bbba-11e8-b2ba-ac1f6b6435d0",
    "affe3674-bb9c-11e8-b2b9-ac1f6b6435d0",
]

CLASS_LABEL_MAP = {
    0:  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',
    7:  'Golgi apparatus',
    8:  'Peroxisomes',
    9:  'Endosomes',
    10:  'Lysosomes',
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',
    14:  'Microtubules',
    15:  'Microtubule ends',
    16:  'Cytokinetic bridge',
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',
    22:  'Cell junctions',
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',
    27:  'Rods & rings'
}

############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 64
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have 28 classes.
        # Naming the dataset nucleus, and the class nucleus
        for k, v in CLASS_LABEL_MAP.items():
            self.add_class(v, k, v)
        # self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)

        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = [f.split(".png")[0] for f in os.listdir(dataset_dir) if f.endswith(".png")]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, "{}.png".format(image_id)))

    def load_mask(self, image_id,channel='blue'):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = info['path'].split(info['id'])[0].replace('dev', 'official').replace('rgb', 'official')

        # Read mask files from .png image
        # Get mask for 0: Nucleoplasm
        img = skimage.io.imread(os.path.join(mask_dir, "{}_"+channel+".png".format(info['id'])))
        mask = get_mask(img,channel)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []

    # print(dataset.image_ids)

    for image_id in dataset.image_ids:
        print(image_id)
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # print("---ROIS---")
        # print(r["rois"])
        # print("---CLASS_IDS---")
        # print(r["class_ids"])
        # print("---SCORES---")
        # print(r["scores"])
        # print("---MASKS---")
        # print(r["masks"])
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        print(rle)
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
