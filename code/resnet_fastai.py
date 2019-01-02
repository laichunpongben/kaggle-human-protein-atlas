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
from fastai.callbacks.tracker import EarlyStoppingCallback, ReduceLROnPlateauCallback
from sklearn.preprocessing import MultiLabelBinarizer

from .utils import open_4_channel
from .arch import Resnet4Channel, Inception4Channel, SqueezeNet4Channel
from .loss import focal_loss, f1_loss
from .callback import SaveModelCustomPathCallback, CSVCustomPathLogger
from .ml_stratifiers import MultilabelStratifiedShuffleSplit
from .train import my_fit_one_cycle
from config import DATASET_PATH, MODEL_PATH, PRED_PATH, OUT_PATH, PLOT_PATH, LOG_PATH, STATS, WEIGHTS, formatter


###############################
# Set training hyperparameters
###############################

parser = argparse.ArgumentParser()
parser.add_argument("-a","--arch", help="Neural network architecture", type=str, choices=["resnet", "inception", "squeezenet"], default="resnet")
parser.add_argument("-b","--batchsize", help="Batch size", type=int, default=64)
parser.add_argument("-d","--encoderdepth", help="Encoder depth of the network", type=int, choices=[18,34,50,101,152], default=50)
parser.add_argument("-D","--dataset", help="Dataset", type=str, choices=["official", "hpav18", "official_hpav18"], default="official")
parser.add_argument("-e","--epochnum1", help="Epoch number for stage 1", type=int, default=0)
parser.add_argument("-E","--epochnum2", help="Epoch number for stage 2", type=int, default=0)
parser.add_argument("-f","--fold", help="K fold cross validation", type=int, default=1)
parser.add_argument("-i","--gpuid", help="GPU device id", type=int, choices=range(-1, 8), default=0)
parser.add_argument("-l","--loss", help="Loss function", type=str, choices=["bce", "focal", "f1"], default="bce")
parser.add_argument("-m","--model", help="Trained model to load", type=str, default=None)
parser.add_argument("-p","--dropout", help="Dropout ratio", type=float, default=0.5)
parser.add_argument("-r","--learningrate", help="Learning rate", type=float, default=0)
parser.add_argument("-s","--imagesize", help="Image size", type=int, default=512)
parser.add_argument("-S","--sampler", help="Sampler", type=str, choices=["random", "weighted"], default="random")
parser.add_argument("-t","--thres", help="Threshold", type=float, default=0.1)
parser.add_argument("-u","--unfreezeto", help="Number of layers to unfreeze", type=int, default=0)
parser.add_argument("-v","--verbose", help="Set verbosity 0-3, 0 to turn off output (not yet implemented)", type=int, default=1)


args = parser.parse_args()
if args.gpuid >= 0:
    device = torch.cuda.set_device(args.gpuid)
    # nvidia_smi.nvmInit()
    # nvi_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
else:
    device = 'cpu'

bs     = args.batchsize
th     = args.thres
ds     = args.dataset
fold   = args.fold
uf     = args.unfreezeto
epochnum1 = args.epochnum1
epochnum2 = args.epochnum2
lr        = args.learningrate


if not args.model:
    dropout   = args.dropout
    imgsize   = args.imagesize
    arch      = args.arch
    enc_depth = args.encoderdepth
    loss      = args.loss
    sampler   = args.sampler
    old_ep1   = 0
    old_ep2   = 0
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
        search = re.search('(bce|focal|f1)', runname)
        return search.group(1) if search else 'bce'

    def get_sampler(runname):
        search = re.search('(random|weighted)', runname)
        return search.group(1) if search else 'random'

    def get_lr(runname):
        search = re.search('-lr(\S+?)-', runname)
        return float(search.group(1)) if search else args.learningrate

    def get_bs(runname):
        search = re.search('-bs(\d+)-', runname)
        return int(search.group(1)) if search else bs

    def get_old_ep1(runname):
        search = re.search('-ep(\d+)_', runname)
        return int(search.group(1)) if search else 0

    def get_old_ep2(runname):
        search = re.search('-ep\d+_(\d+)', runname)
        return int(search.group(1)) if search else 0

    runname   = re.sub('stage-[12]-', '', str(Path(args.model).name))
    dropout   = float(re.search('-drop(0.\d+)',runname).group(1))
    imgsize   = int(re.search('(?<=resnet).+?-(\d+)', runname).group(1))
    arch      = re.search('^(\D+)', runname).group(1)
    loss      = get_loss(runname)
    sampler   = get_sampler(runname)
    enc_depth = int(re.search('^\D+(\d+)', runname).group(1))
    old_ep1   = get_old_ep1(runname)
    old_ep2   = get_old_ep2(runname)


num_class = 28
# mean and std in of each channel in the train set
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

src_path = Path(DATASET_PATH)
src_path.mkdir(parents=True, exist_ok=True)

out_path = Path(OUT_PATH)
out_path.mkdir(parents=True, exist_ok=True)

model_path = Path(MODEL_PATH)
model_path.mkdir(parents=True, exist_ok=True)

pred_path = Path(PRED_PATH)
pred_path.mkdir(parents=True, exist_ok=True)

plot_path = Path(PLOT_PATH)
plot_path.mkdir(parents=True, exist_ok=True)

log_path = Path(LOG_PATH)
plot_path.mkdir(parents=True, exist_ok=True)

train_csv = src_path/f'train.csv'
hpav18_csv = src_path/f'train_hpa_no_uncertain.csv'

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
                    'Learning rate #1: ' + str(lr),
                    'Batch size: ' + str(bs),
                    'Dataset: ' + str(ds),
                    'Dataset directory: ' + str(src_path),
                    'Output directory: ' + str(out_path)
                    ])
logger.debug("Start a new training task")
logger.info(conf_msg)

# def get_gpu_stats():
#     res = nvidia_smi.nvmlDeviceGetUtilizationRates(nvi_handle)
#     print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
#     return


###############################
# Load & preprocess data
###############################

np.random.seed(42)

stats = STATS["official"]
logger.info("{} stats: {}".format(ds, stats))
logger.info("official stats: {}".format(STATS["official"]))
logger.info("hpav18 stats: {}".format(STATS["hpav18"]))

test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(src_path/'test') if fname.endswith('.png')}))
logger.debug("# Test ids: {}".format(len(test_ids)))
test_fnames = [src_path/'test'/test_id for test_id in test_ids]

def extract_rare(df):
    lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]  # 0.520
    # lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27,12,13,16,18,20,22]
    df_orig = df.copy()
    df = df[0:0]
    logger.debug("empty size {}".format(df.shape))
    for i in lows:
        target = str(i)
        indicies = df_orig.loc[df_orig['Target'] == target].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.startswith(target+" ")].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.endswith(" "+target)].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.contains(" "+target+" ")].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
    return df

def oversample_df(df):
    lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]
    df_orig = df.copy()
    for i in lows:
        target = str(i)
        indicies = df_orig.loc[df_orig['Target'] == target].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.startswith(target+" ")].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.endswith(" "+target)].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.contains(" "+target+" ")].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
    return df

train_df = pd.read_csv(train_csv)
logger.debug("official train size: {}".format(train_df.shape))

train_df = oversample_df(train_df)
logger.debug("oversample official size: {}".format(train_df.shape))

hpav18_df = pd.read_csv(hpav18_csv)
logger.debug("hpav18 train size: {}".format(hpav18_df.shape))

hpav18_df = extract_rare(hpav18_df)
logger.debug("hpav18 rare size: {}".format(hpav18_df.shape))

train_df = pd.concat([train_df, hpav18_df], ignore_index=True)
logger.debug("concat size: {}".format(train_df.shape))

def generate_train_valid_split(df, n_splits=3, valid_size=0.2):
    X, y = df.Id, df.Target
    y = MultiLabelBinarizer().fit_transform(y)
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=valid_size, random_state=42)
    return msss.split(X, y)

def get_src(valid_idx=None, split_pct=0.2):
    src = ImageItemList.from_df(train_df, path=src_path, folder='train', suffix='.png')
    if valid_idx is not None:
        src = src.split_by_idx(valid_idx)
    else:
        src = src.random_split_by_pct(split_pct)
    src = src.label_from_df(sep=' ',  classes=[str(i) for i in range(num_class)])
    return src

inv_weights = torch.FloatTensor([1/w for w in WEIGHTS])
if device != 'cpu':
    inv_weights = inv_weights.cuda()

def get_multilabel_weights(dl):
    weights = []
    for index, (x,y) in enumerate(dl):
        w, _ = torch.max(y.mul(inv_weights), dim=1)
        w = w.tolist()
        weights.extend(w)
    return weights

def get_data(src, is_normalize=True):
    src.train.x.create_func = open_4_channel
    src.train.x.open = open_4_channel
    src.valid.x.create_func = open_4_channel
    src.valid.x.open = open_4_channel

    logger.debug(src.train)
    logger.debug(src.valid)

    src.add_test(test_fnames, label='0')
    src.test.x.create_func = open_4_channel
    src.test.x.open = open_4_channel

    trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1.5,
                                max_lighting=0.05, max_warp=0.)
    data = (src.transform((trn_tfms, _), size=imgsize)
            .databunch(bs=bs))
    if is_normalize:
        data = data.normalize(stats)

    logger.debug("Databunch created")

    if sampler == 'weighted':
        dls = [data.train_dl, data.valid_dl, data.test_dl]
        for dl in dls:
            weights = torch.DoubleTensor(get_multilabel_weights(dl))
            logger.debug("Initialising WeightedRandomSampler with {} weights.".format(len(weights)))
            weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            dl.sampler = weighted_sampler
        logger.debug("Use weighted random sampler")

    return data


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
        "f1": f1_loss,
        "bce": F.binary_cross_entropy_with_logits
    }
    return losses.get(loss, F.binary_cross_entropy_with_logits)


def _prep_model(data, fold=0):
    logger.info('Initialising model.')

    arch_func = get_arch_func(arch)
    loss_func = get_loss_func(loss)
    split = get_split(arch)

    f1_score = partial(fbeta, thresh=0.2, beta=1)
    early_stopping_callback = partial(EarlyStoppingCallback,
                                      # monitor='fbeta',
                                      min_delta=0,
                                      patience=5)
    reduce_lr_on_plateau_callback = partial(ReduceLROnPlateauCallback,
                                            # monitor='fbeta',
                                            min_delta=0.001,
                                            factor=0.2,
                                            patience=5)
    csv_logger = partial(CSVCustomPathLogger,
                         filename="{}-{}".format(runname, fold))

    save_model_callback = partial(SaveModelCustomPathCallback,
                                  # monitor='fbeta',
                                  mode='auto',
                                  every='improvement',
                                  name='{}-{}'.format(runname, fold),
                                  device=device)

    cb = [
        early_stopping_callback,
        reduce_lr_on_plateau_callback,
        csv_logger
    ]
    # if args.gpuid > 0:
    #     cb.append(save_model_callback)

    learn = create_cnn(
                        data,
                        arch_func,
                        cut=-2,
                        split_on=split,
                        ps=dropout,
                        loss_func=loss_func,
                        path=src_path,
                        metrics=[f1_score],
                        callback_fns=cb
                      )
    logger.info('Complete initialising model.')
    return learn

###############################
# Fit model
###############################

def fit_model(learn, stage=1, fold=0):
    global lr
    assert stage in [1, 2]

    if stage == 2:
        learn.freeze_to(-uf)
        logger.debug("Unfreezing model")

    if args.learningrate == 0:
        if stage == 1:
            start_lr = 0.01
            end_lr = 0.04
            num_it = 1000
        else:
            start_lr = 8e-6
            end_lr = 3e-5
            num_it = 1000

        logger.debug("Start finding LR")
        learn.lr_find(start_lr=start_lr, end_lr=end_lr, num_it=num_it)
        lr_curve = list(zip(learn.recorder.lrs, learn.recorder.losses))
        logger.debug(lr_curve)
        best_lr = min(lr_curve, key=lambda x: x[1].data)[0]
        logger.debug("Best LR: {}".format(best_lr))
        learn.recorder.plot()
        plt.show()
        plt.savefig(plot_path/f'stage-{stage}-{runname}-{fold}.png')

        logger.info('Start model fitting: Stage {}'.format(stage))

        factor = 0.7  # arbitrary
        lr = best_lr * factor

    logger.debug("LR: {}".format(lr))

    if stage == 1:
        cyc_len = epochnum1
        max_lr = slice(lr)
    else:
        cyc_len = epochnum2
        max_lr = slice(lr, lr*2e2)

    learn.__class__.fit_one_cycle = my_fit_one_cycle

    learn.fit_one_cycle(cyc_len, max_lr)
    logger.info('Complete model fitting: Stage {}'.format(stage))

    path = model_path/f'stage-{stage}-{runname}-{fold}.pth'
    torch.save(learn.model.state_dict(), path)
    logger.info('Stage {} model saved.'.format(stage))

    return learn

###############################
# Predict
###############################

def _predict(learn, fold=0):
    logger.info('Start predicting test set')
    preds,_ = learn.get_preds(DatasetType.Test)
    logger.info('Complete test prediction.')

    path = pred_path/f'{runname}-{fold}.pth'
    torch.save(preds, path)
    logger.info('Prediction saved.')

    return preds

###############################
# Output results
###############################

def _output_results(preds, suffix=""):
    pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>th)[0]])) for row in np.array(preds)]
    df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
    out_file = out_path/f'{runname}{suffix}.csv'
    df.to_csv(out_file, header=True, index=False)
    logger.info('Results written to {}. Finished! :)'.format(out_file))
    return


if __name__=='__main__':
    all_preds = []
    train_valid_split = generate_train_valid_split(train_df, n_splits=fold, valid_size=0.2)
    for index, (train_idx, valid_idx) in enumerate(train_valid_split):
        if index in [0, 1]:
            continue
        # index = 0
        # src = get_src()
        logger.debug("Start of fold {}".format(index))
        logger.debug("Size of valid set: {}".format(valid_idx.shape[0]))
        src = get_src(valid_idx=valid_idx)
        data = get_data(src)
        learn = _prep_model(data, fold=index)

        if args.model:
            logger.debug("runname: {}".format(runname))
            logger.info('Loading model: {}, with suffix {}'.format(args.model, index))
            path = model_path/f'{args.model}-{index}.pth'
            learn.model.load_state_dict(torch.load(path,
                                                   map_location=device),
                                        strict=False)
            logger.info('Finish loading model.')
        else:
            logger.info('No pretrained model.')

        if epochnum1 + epochnum2 > 0:
            if epochnum1 > 0:
                learn = fit_model(learn, stage=1, fold=index)
            if epochnum2 > 0:
                learn = fit_model(learn, stage=2, fold=index)
        else:
            assert args.model is not None

        preds = _predict(learn, fold=index)
        all_preds.append(preds)
        _output_results(preds, suffix="-{}".format(index))

    all_preds = torch.stack(all_preds)

    # TODO: implement ensemble here

    avg_preds = torch.mean(all_preds, dim=0)
    logger.debug(avg_preds.shape)

    if fold > 1:
        _output_results(avg_preds, suffix="-avg")
