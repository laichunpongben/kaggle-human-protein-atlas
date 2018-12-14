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

bs = 64
size = 224
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

def resnet50(pretrained):
    return Resnet4Channel(encoder_depth=50)

# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6],m[1])

f1_score = partial(fbeta, thresh=0.2, beta=1)

learn = create_cnn(
    data,
    resnet50,
    cut=-2,
    split_on=_resnet_split,
    loss_func=F.binary_cross_entropy_with_logits,
    path=src_path,
    metrics=[f1_score],
)

# learn.lr_find()
# learn.recorder.plot()
lr = 3e-2
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50-datablocks')
learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()
learn.fit_one_cycle(15, slice(3e-5, lr/5))
learn.save('stage-2-rn50')
preds,_ = learn.get_preds(DatasetType.Test)
pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.1)[0]])) for row in np.array(preds)]
df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
df.to_csv(out_path/'resnet50_32_0.csv', header=True, index=False)
