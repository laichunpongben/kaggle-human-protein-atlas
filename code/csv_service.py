from collections import defaultdict
import pandas as pd


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

if __name__ == '__main__':

    csvs = [
        "output/resnet50-512-bce-random-drop0.5-th0.1-bs16-lr0.01-ep15_25.csv",  # 0.465
        "output/resnet50-512-official_hpav18-bce-weighted-drop0.5-th0.1-bs16-lr0.005-ep5_15.csv",  # 0.465
        "output/resnet50-512-official-bce-weighted-drop0.5-th0.1-bs16-lr0.005-ep5_15.csv",  # 0.460
        "output/resnet50-512-bce-random-drop0.5-th0.1-bs16-lr0.01-ep5_15.csv",  # 0.458
        "output/resnet50-512-bce-weighted-drop0.5-th0.1-bs16-lr0.01-ep5_15.csv"  # 0.455
    ]
    out = "output/ensemble_0_1_2_3_4_vote4_best2.csv"
    min_vote = 4
    first_n = 2
    ensemble(csvs, out, min_vote, first_n)
