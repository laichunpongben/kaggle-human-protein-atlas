# Adopted from https://github.com/wdhorton/protein-atlas-fastai

import os
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from fastai import *
from fastai.vision import *

from .utils import open_4_channel
from .resnet import Resnet4Channel
from .loss import focal_loss
from config import DATASET_PATH, OUT_PATH, formatter
import argparse
import logging
import datetime


###############################
# Set training config
###############################

parser = argparse.ArgumentParser()
parser.add_argument("-a","--arch", help="Neural network architecture (only resnet for now)", type=str, choices=["resnet"], default="resnet")
parser.add_argument("-b","--batchsize", help="batch size (not in use yet)", type=int, default=64)
parser.add_argument("-d","--encoderdepth", help="encoder depth of the network", type=int, choices=[34,50,101,152], default=152)
parser.add_argument("-e","--epochnum1", help="epoch number for stage 1", type=int, default=25)
parser.add_argument("-E","--epochnum2", help="epoch number for stage 2", type=int, default=50)
parser.add_argument("-i","--gpuid", help="GPU device id", type=int, choices=range(8), default=0)
parser.add_argument("-l","--loss", help="Loss function", type=str, choices=["bce", "focal"], default="bce")
parser.add_argument("-m","--model", help="Path to retrained model to load", type=str, default=None)
parser.add_argument("-p","--dropout", help="dropout (float)", type=float, default=0.5)
parser.add_argument("-s","--imagesize", help="image size", type=int, default=256)
parser.add_argument("-t","--thres", help="threshold", type=float, default=0.1)
parser.add_argument("-v","--verbosity", help="set verbosity 0-3, 0 to turn off output (not yet implemented)", type=int, default=1)

args = parser.parse_args()

if args.gpuid>=0:
    torch.cuda.set_device(args.gpuid)

device = torch.cuda.set_device(args.gpuid)
bs     = args.batchsize
th     = args.thres
loss   = args.loss

if not args.model:
    dropout   = args.dropout
    imgsize   = args.imagesize
    arch      = args.arch
    enc_depth = args.encoderdepth
    epochnum1 = args.epochnum1
    epochnum2 = args.epochnum2
    runname   = arch+str(args.encoderdepth)+'-'+str(imgsize)+'-drop'+str(dropout)+'-ep'+str(args.epochnum1)+'_'+str(args.epochnum2)
else:
    runname   = str(Path(args.model).stem)
    dropout   = float(re.search('-drop(0.\d+)',runname).group(1))
    imgsize   = int(re.search('-(\d+)', runname).group(1))
    arch      = re.search('^stage-[12]-(\D+)', runname).group(1)
    enc_depth = int(re.search('^stage-[12]-\D+(\d+)', runname).group(1))
    epochnum1 = int(re.search('-ep(\d+)_', runname).group(1))
    epochnum2 = int(re.search('-ep\d+_(\d+)', runname).group(1))

runname = (arch +
          str(args.encoderdepth) +
          '-' + str(imgsize) +
          '-' + str(loss) +
          '-drop' + str(dropout) +
          '-ep' + str(args.epochnum1) +
          '_' + str(args.epochnum2))

num_class = 28
# mean and std in of each channel in the train set
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
protein_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])

src_path = Path(DATASET_PATH)
out_path = Path(OUT_PATH)


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
                    'Encoder depth: ' + str(enc_depth),
                    'Dropout: ' + str(dropout),
                    'Threshold: ' + str(th),
                    'Stage 1 #epoch: ' + str(epochnum1),
                    'Stage 2 #epoch: ' + str(epochnum2),
                    'Dataset directory: ' + str(src_path),
                    'Output directory: ' + str(out_path)
               ])
logger.debug("Start a new training task")
logger.info(conf_msg)

###############################
# Load & preprocess data
###############################

np.random.seed(42)
src = (ImageItemList.from_csv(src_path, 'train.csv', folder='train', suffix='.png')
       .random_split_by_pct(0.2)
       .label_from_df(sep=' ',  classes=[str(i) for i in range(num_class)]))

src.train.x.create_func = open_4_channel
src.train.x.open = open_4_channel
src.valid.x.create_func = open_4_channel
src.valid.x.open = open_4_channel

test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(src_path/'test') if fname.endswith('.png')}))
test_fnames = [src_path/'test'/test_id for test_id in test_ids]
src.add_test(test_fnames, label='0')
src.test.x.create_func = open_4_channel
src.test.x.open = open_4_channel

trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1,
                            max_lighting=0.05, max_warp=0.)
data = (src.transform((trn_tfms, _), size=imgsize)
        .databunch().normalize(protein_stats))

###############################
# Set up model
###############################

def resnet(pretrained):
    return Resnet4Channel(encoder_depth=enc_depth)

# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6],m[1])

def _prep_model():
    logger.info('Initialising model.')
    losses = {
         "focal": focal_loss,
        "bce": F.binary_cross_entropy_with_logits
    }
    loss_func = losses.get(loss, F.binary_cross_entropy_with_logits)

    f1_score = partial(fbeta, thresh=0.2, beta=1)
    learn = create_cnn(
                        data,
                        resnet,
                        cut=-2,
                        split_on=_resnet_split,
                        ps=dropout,
                        loss_func=loss_func,
                        path=src_path,
                        metrics=[f1_score],
                      )
    logger.info('Complete initialising model.')
    return learn

###############################
# Fit model
###############################

def _fit_model(learn):
    # learn.lr_find()
    # learn.recorder.plot()
    lr = 3e-2
    logger.info('Start model fitting: Stage 1')
    learn.fit_one_cycle(epochnum1, slice(lr))
    learn.save('stage-1-'+runname)
    logger.info('Complete model fitting Stage 1. Model saved.')
    learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot()
    logger.info('Start model fitting: Stage 2')
    learn.fit_one_cycle(epochnum2, slice(3e-5, lr/epochnum2))
    learn.save('stage-2-'+runname)
    logger.info('Complete model fitting Stage 2. Model saved.')
    return learn

###############################
# Predict
###############################

def _predict(learn):
    preds,_ = learn.get_preds(DatasetType.Test)
    logger.info('Complete model fitting.')
    return preds

###############################
# Output results
###############################

def _output_results(preds):
    pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>th)[0]])) for row in np.array(preds)]
    df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
    df.to_csv(OUT_PATH+runname+'.csv', header=True, index=False)
    logger.info('Results written to file. Finshed! :)')
    return


if __name__=='__main__':
    learn = _prep_model()
    if not args.model:
        learn = _fit_model(learn)
    else:
        logger.debug(runname)
        logger.info('Loading model: '+args.model)
        #learn = learn.load_state_dict(torch.load(args.model))
        #learn = learn.to(device)
        state_dict = torch.load(args.model)
        logger.debug(state_dict['model'].keys())
        logger.debug('hello')
        learn.model.load_state_dict(state_dict['model'])
    preds = _predict(learn)
    _output_results(preds)
