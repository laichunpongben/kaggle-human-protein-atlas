# Adopted from https://github.com/wdhorton/protein-atlas-fastai

import os
from pathlib import Path
import argparse
import logging
import time
import nvidia_smi

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fastai import *
from fastai.vision import *
from fastai.callbacks.tracker import EarlyStoppingCallback
from sklearn.preprocessing import MultiLabelBinarizer

from .utils import open_4_channel
from .arch import Resnet4Channel, Inception4Channel, SqueezeNet4Channel
from .loss import focal_loss
from .callback import SaveModelCustomPathCallback, CSVCustomPathLogger
from .ml_stratifiers import MultilabelStratifiedShuffleSplit
from config import DATASET_PATH, MODEL_PATH, OUT_PATH, STATS, WEIGHTS, formatter


###############################
# Set training hyperparameters
###############################

parser = argparse.ArgumentParser()
parser.add_argument("-a","--arch", help="Neural network architecture", type=str, choices=["resnet", "inception", "squeezenet"], default="resnet")
parser.add_argument("-b","--batchsize", help="batch size", type=int, default=64)
parser.add_argument("-d","--encoderdepth", help="encoder depth of the network", type=int, choices=[34,50,101,152], default=152)
parser.add_argument("-D","--dataset", help="Dataset", type=str, choices=["official", "hpav18", "official_hpav18"], default="official")
parser.add_argument("-e","--epochnum1", help="epoch number for stage 1", type=int, default=5)
parser.add_argument("-E","--epochnum2", help="epoch number for stage 2", type=int, default=15)
parser.add_argument("-i","--gpuid", help="GPU device id", type=int, choices=range(-1, 8), default=0)
parser.add_argument("-l","--loss", help="loss function", type=str, choices=["bce", "focal"], default="bce")
parser.add_argument("-m","--model", help="trained model to load", type=str, default=None)
parser.add_argument("-p","--dropout", help="dropout ratio", type=float, default=0.5)
parser.add_argument("-r","--learningrate", help="learning rate", type=float, default=3e-2)
parser.add_argument("-s","--imagesize", help="image size", type=int, default=256)
parser.add_argument("-S","--sampler", help="sampler", type=str, choices=["random", "weighted"], default="random")
parser.add_argument("-t","--thres", help="threshold", type=float, default=0.1)
parser.add_argument("-v","--verbose", help="set verbosity 0-3, 0 to turn off output (not yet implemented)", type=int, default=1)


args = parser.parse_args()
if args.gpuid >= 0:
    device = torch.cuda.set_device(args.gpuid)
    nvidia_smi.nvmInit()
    nvi_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)

else:
    device = 'cpu'
bs     = args.batchsize
th     = args.thres
ds     = args.dataset

if not args.model:
    dropout   = args.dropout
    imgsize   = args.imagesize
    arch      = args.arch
    enc_depth = args.encoderdepth
    loss      = args.loss
    sampler   = args.sampler
    lr        = args.learningrate
    epochnum1 = args.epochnum1
    epochnum2 = args.epochnum2
    runname = (arch +
              str(args.encoderdepth) +
              '-' + str(imgsize) +
              '-' + str(ds) +
              '-' + str(loss) +
              '-' + str(sampler) +
              '-drop' + str(dropout) +
              '-th' + str(th) +
              '-bs' + str(bs) +
              '-lr' + str(lr) +
              '-ep' + str(args.epochnum1) +
              '_' + str(args.epochnum2))
else:
    def get_loss(runname):
        search = re.search('(bce|focal)', runname)
        return search.group(1) if search else 'bce'

    def get_sampler(runname):
        search = re.search('(random|weighted)', runname)
        return search.group(1) if search else 'random'

    def get_lr(runname):
        search = re.search('-lr(\S+)-', runname)
        return search.group(1) if search else args.learningrate

    def get_bs(runname):
        search = re.search('-bs(\d+)-', runname)
        return search.group(1) if search else bs

    runname   = re.sub('stage-[12]-', '', str(Path(args.model).name))
    dropout   = float(re.search('-drop(0.\d+)',runname).group(1))
    imgsize   = int(re.search('(?<=resnet).+?-(\d+)', runname).group(1))
    arch      = re.search('^(\D+)', runname).group(1)
    loss      = get_loss(runname)
    lr        = get_lr(runname)
    sampler   = get_sampler(runname)
    enc_depth = int(re.search('^\D+(\d+)', runname).group(1))
    epochnum1 = int(re.search('-ep(\d+)_', runname).group(1))
    epochnum2 = int(re.search('-ep\d+_(\d+)', runname).group(1))

num_class = 28
# mean and std in of each channel in the train set
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

src_path = Path(DATASET_PATH)
out_path = Path(OUT_PATH)
train_csv = src_path/f'train.csv'

###############################
# Set up logger
###############################

logger = logging.getLogger("code.resnet_fastai")
file_handler = logging.FileHandler('logs/resnet_fastai.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(level=logging.DEBUG)

conf_msg = '\n'.join([
                    'Device ID: ' + str(args.gpuid),
                    'Image size: ' + str(imgsize),
                    'Network architecture: ' + str(arch),
                    'Loss function: ' + str(loss),
                    'Sampler: ' + str(sampler),
                    'Encoder depth: ' + str(enc_depth),
                    'Dropout: ' + str(dropout),
                    'Threshold: ' + str(th),
                    'Stage 1 #epoch: ' + str(epochnum1),
                    'Stage 2 #epoch: ' + str(epochnum2),
                    'Batch size: ' + str(bs),
                    'Dataset: ' + str(ds),
                    'Dataset directory: ' + str(src_path),
                    'Output directory: ' + str(out_path)
                    ])
logger.debug("Start a new training task")
logger.info(conf_msg)

def get_gpu_stats():
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(nvi_handle)
    print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    return


###############################
# Load & preprocess data
###############################

np.random.seed(42)

def generate_train_valid_split(train_csv, n_splits=10, valid_size=0.2):
    df = pd.read_csv(train_csv)
    X, y = df.Id, df.Target
    y = MultiLabelBinarizer().fit_transform(y)
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=valid_size, random_state=42)
    return msss.split(X, y)

train_valid_split = generate_train_valid_split(train_csv, n_splits=10, valid_size=0.2)


src = (ImageItemList.from_csv(src_path, 'train.csv', folder='train', suffix='.png')
       .random_split_by_pct(0.2)
       .label_from_df(sep=' ',  classes=[str(i) for i in range(num_class)]))

if "official" in ds:
    logger.info("Offical stats: {}".format(STATS["official"]))
if "hpav18" in ds:
    logger.info("HPAv18 stats: {}".format(STATS["hpav18"]))

src.train.x.create_func = open_4_channel
src.train.x.open = open_4_channel
src.valid.x.create_func = open_4_channel
src.valid.x.open = open_4_channel


test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(src_path/'test') if fname.endswith('.png')}))
logger.debug("# Test ids: ".format(len(test_ids)))
test_fnames = [src_path/'test'/test_id for test_id in test_ids]
src.add_test(test_fnames, label='0')
src.test.x.create_func = open_4_channel
src.test.x.open = open_4_channel

trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1,
                            max_lighting=0.05, max_warp=0.)
data = (src.transform((trn_tfms, _), size=imgsize)
        .databunch(bs=bs))

logger.debug("Databunch created")

def sort_class_by_rarity(weights):
    return [y for y,_ in sorted(list(zip(range(num_class), weights)), key=lambda x: x[1])]

sorted_class = sort_class_by_rarity(WEIGHTS)

def get_rarest_class_weight(y):
    max_w = 1e9
    weights = []
    for row in y:
        hasLabel = False
        for c in sorted_class:
            if row[c]:
                try:
                    w = 1/WEIGHTS[c]  # invert the weights
                except ZeroDivisionError:
                    w = max_w
                weights.append(w)
                hasLabel = True
                break
        if not hasLabel:
            try:
                w = 1/WEIGHTS[sorted_class[-1]]
            except ZeroDivisionError:
                w = max_w
            weights.append(w)

    return weights

def get_multilabel_weights(data):
    weights = []
    for x,y in iter(data.train_dl):
        weights.extend(get_rarest_class_weight(y))
    return weights

if sampler == 'weighted':
    weights = get_multilabel_weights(data)
    logger.debug("Initialising WeightedRandomSampler with {} weights.".format(len(weights)))
    weights = torch.DoubleTensor(weights)
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data.train_dl.sampler = weighted_sampler
    data.test_dl.sampler = weighted_sampler


###############################
# Set up model
###############################

def resnet(pretrained):
    return Resnet4Channel(encoder_depth=enc_depth)

def inception(pretrained):
    return Inception4Channel()

def squeezenet(pretrained):
    return SqueezeNet4Channel()

def get_arch_func(arch):
    archs = {
        "resnet": resnet,
        "inception": inception,
        "sequeezenet": squeezenet
    }
    return archs.get(arch, resnet)

# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6],m[1])

def _default_split(m): return (m[1],)

def get_split(arch):
    if arch in ['resnet']:
        return _resnet_split
    else:
        return _default_split

def get_loss_func(loss):
    losses = {
        "focal": focal_loss,
        "bce": F.binary_cross_entropy_with_logits
    }
    return losses.get(loss, F.binary_cross_entropy_with_logits)


def _prep_model():
    logger.info('Initialising model.')

    arch_func = get_arch_func(arch)
    loss_func = get_loss_func(loss)
    split = get_split(arch)

    f1_score = partial(fbeta, thresh=0.2, beta=1)
    early_stopping_callback = partial(EarlyStoppingCallback,
                                      monitor='fbeta',
                                      min_delta=0.005,
                                      patience=3)
    csv_logger = partial(CSVCustomPathLogger,
                         filename=runname)

    # TODO: Fix OSError: [Errno 12] Cannot allocate memory
    # save_model_callback = partial(SaveModelCustomPathCallback,
    #                               monitor='fbeta',
    #                               mode='auto',
    #                               every='improvement',
    #                               name=runname+'-bestmodel',
    #                               device=device)

    learn = create_cnn(
                        data,
                        arch_func,
                        cut=-2,
                        split_on=split,
                        ps=dropout,
                        loss_func=loss_func,
                        path=src_path,
                        metrics=[f1_score],
                        callback_fns=[
                            early_stopping_callback,
                            csv_logger,
                            # save_model_callback
                        ]
                      )
    logger.info('Complete initialising model.')
    return learn

###############################
# Fit model
###############################

def _fit_model(learn):
    # learn.lr_find()
    # learn.recorder.plot()
    logger.info('Start model fitting: Stage 1')
    learn.fit_one_cycle(epochnum1, slice(lr))

    stage1_model_path = Path(MODEL_PATH)/f'stage-1-{runname}.pth'
    logger.info('Complete model fitting Stage 1.')
    torch.save(learn.model.state_dict(), stage1_model_path)
    logger.info('Model saved.')

    learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot()
    logger.info('Start model fitting: Stage 2')
    learn.fit_one_cycle(epochnum2, slice(lr*e-3, lr/epochnum2))

    stage2_model_path = Path(MODEL_PATH)/f'stage-2-{runname}.pth'
    logger.info('Complete model fitting Stage 2.')
    torch.save(learn.model.state_dict(), stage2_model_path)
    logger.info('Model saved.')

    return learn

###############################
# Predict
###############################

def _predict(learn):
    preds,_ = learn.get_preds(DatasetType.Test)
    logger.info('Complete test prediction.')
    return preds

###############################
# Output results
###############################

def _output_results(preds):
    pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>th)[0]])) for row in np.array(preds)]
    df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
    out_file = OUT_PATH+runname+'.csv'
    df.to_csv(out_file, header=True, index=False)
    logger.info('Results written to {}. Finished! :)'.format(out_file))
    return


if __name__=='__main__':
    learn = _prep_model()
    if not args.model:
        learn = _fit_model(learn)
    else:
        logger.debug(runname)
        logger.info('Loading model: '+args.model)
        model_path = Path(MODEL_PATH)/f'{args.model}.pth'
        learn.model.load_state_dict(torch.load(model_path,
                                               map_location=device),
                                    strict=False)
    preds = _predict(learn)
    _output_results(preds)
