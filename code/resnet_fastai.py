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
from config import DATASET_PATH, OUT_PATH
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--gpuid", help="GPU device id", type=int, choices=range(8), default=0)
parser.add_argument("-s","--imagesize", help="image size", type=int, default=256)
parser.add_argument("-b","--batchsize", help="batch size (not in use yet)", type=int, default=64)
parser.add_argument("-d","--encoderdepth", help="encoder depth of the network", type=int, choices=[34,50,101,152], default=152)
parser.add_argument("-p","--dropout", help="dropout (float)", type=float, default=0.5)
parser.add_argument("-e","--epochnum1", help="epoch number for stage 1", type=int, default=25)
parser.add_argument("-E","--epochnum2", help="epoch number for stage 2", type=int, default=50)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
bs = args.batchsize
dropout = args.dropout
size = args.imagesize
runname='rn'+str(args.encoderdepth)+'-'+str(size)+'-drop'+str(dropout)+'-ep'+str(args.epochnum1)+'_'+str(args.epochnum2)


num_class = 28
protein_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
src_path = Path(DATASET_PATH)
out_path = Path(OUT_PATH)

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
data = (src.transform((trn_tfms, _), size=size)
        .databunch().normalize(protein_stats))

def resnet(pretrained):
    return Resnet4Channel(encoder_depth=args.encoderdepth)

# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6],m[1])

f1_score = partial(fbeta, thresh=0.2, beta=1)

learn = create_cnn(
    data,
    resnet,
    cut=-2,
    split_on=_resnet_split,
    ps=dropout,
    loss_func=F.binary_cross_entropy_with_logits,
    path=src_path,
    metrics=[f1_score],
)

# learn.lr_find()
# learn.recorder.plot()
lr = 3e-2
learn.fit_one_cycle(args.epochnum1, slice(lr))
learn.save('stage-1-'+runname)
learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()
learn.fit_one_cycle(args.epochnum2, slice(3e-5, lr/args.epochnum2))
learn.save('stage-2-'+runname)
preds,_ = learn.get_preds(DatasetType.Test)
pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.1)[0]])) for row in np.array(preds)]
df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
df.to_csv(out_path+runname+'.csv', header=True, index=False)
