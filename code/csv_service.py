import os
from collections import defaultdict
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from config import PRED_PATH, OUT_PATH, DATASET_PATH


def get_annotations(train_csv):
    df = pd.read_csv(train_csv, index_col=0)
    columns = df.columns
    annotations = df.to_dict(orient="index")
    annotations = {k: [int(x) for x in sorted(v[columns[-1]].split())] for k, v in annotations.items()}
    return annotations

def get_nuclei_count_density():
    '''
    fname: (count, density)
    '''
    # TODO: implement
    return {}
    # df = pd.read_csv("nuclei.csv", index_col=0)
    # columns = df.columns
    # nuclei_count_density = df.to_dict(orient="index")
    # nuclei_count_density = {k: (int(v[columns[1]]), int(v[columns[2]])) for k, v in nuclei_count_density.items()}
    # return nuclei_count_density


def make_union(csv0, csv1, out, mode="u"):
    df0 = pd.read_csv(csv0, index_col=0)
    df1 = pd.read_csv(csv1, index_col=0)

    ids = df0.index.tolist()
    print(len(ids))

    labels0 = df0.to_dict(orient="index")
    labels0 = {k: [x for x in sorted(str(v[df0.columns[-1]]).split())] for k, v in labels0.items()}

    labels1 = df1.to_dict(orient="index")
    labels1 = {k: [x for x in sorted(str(v[df1.columns[-1]]).split())] for k, v in labels1.items()}

    if mode == "u":
        labels = [' '.join(sorted(list(set(labels0[id_]).union(set(labels1[id_]))))) for id_ in ids]
    else:
        labels = [' '.join(sorted(list(set(labels0[id_]).intersection(set(labels1[id_]))))) for id_ in ids]

    df = pd.DataFrame({'Id':ids,'Predicted':labels})
    print(df)
    df.to_csv(out, header=True, index=False)

def ensemble(csvs, out, min_vote, first_n):
    dfs = [pd.read_csv(csv_, index_col=0) for csv_ in csvs]
    all_label_lists = defaultdict(list)
    all_labels = defaultdict(str)

    for i, df in enumerate(dfs):
        labels = df.to_dict(orient="index")
        for k, v in labels.items():
            all_label_lists[k].append([x for x in sorted(str(v[df.columns[-1]]).split())])

    for k, v in all_label_lists.items():
        all_labels[k] = vote(v, min_vote, first_n)

    print(all_labels)
    ids = list(all_labels.keys())
    labels = list(all_labels.values())

    out_df = pd.DataFrame({'Id':ids,'Predicted':labels})
    out_df.to_csv(out, header=True, index=False)

def vote(list_, min_vote, first_n):
    # [['21', '25'], ['11', '21', '25'], ['21', '23', '25'], ['21', '25'], ['21', '23', '25']]
    d = defaultdict(int)
    for x in list_:
        for c in x:
            d[c] += 1

    labels = sorted([k for k, v in d.items() if v>=min_vote], key=lambda x: -d[x])[:first_n]
    return ' '.join(labels)

def get_preds(runname, fold, start=0):
    assert fold > 0

    all_preds = []
    for i in range(fold):
        if i < start:
            continue
        path = Path(PRED_PATH)/f'{runname}-{i}.pth'
        try:
            preds = torch.load(path)
            all_preds.append(preds)
        except FileNotFoundError as e:
            print(e)
    print("# preds: {}".format(len(all_preds)))
    return all_preds

def output_csv_avg(runname, fold, th):
    preds = get_preds(runname, fold)
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>th)[0]])) for row in np.array(preds)]

    test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(Path(DATASET_PATH)/'test') if fname.endswith('.png')}))
    df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
    out_file = Path(OUT_PATH)/f'{runname}-th{th}-avg.csv'
    df.to_csv(out_file, header=True, index=False)

def output_csv_vote(runname, fold, th, min_vote, first_n=99, start=0):
    test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(Path(DATASET_PATH)/'test') if fname.endswith('.png')}))

    preds = get_preds(runname, fold, start=start)
    csvs = []
    for i in range(fold):
        if i < start:
            continue
        pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>th)[0]])) for row in np.array(preds[i-start])]
        df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
        out_file = Path(OUT_PATH)/f'{runname}-th{th}-{i}.csv'
        csvs.append(out_file)
        df.to_csv(out_file, header=True, index=False)

    vote_out = Path(OUT_PATH)/f'{runname}-vote{min_vote}-th{th}.csv'
    ensemble(csvs, vote_out, min_vote, first_n)

if __name__ == '__main__':
    # runname = "resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_30"
    # fold = 5
    # start = 4
    # th = 0.30
    # min_vote = 0
    # output_csv_vote(runname, fold, th, min_vote, start=start)
    # output_csv_avg(runname, fold, th)

    csvs = [
        "output/resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_30-th0.2-4.csv",  # 0.465
        "output/resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_30-4.csv",  # 0.465
        "output/resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_30-0_1.csv",  # 0.460
    ]
    out = "output/ensemble_537_vote3.csv"
    min_vote = 3
    first_n = 99
    ensemble(csvs, out, min_vote, first_n)
