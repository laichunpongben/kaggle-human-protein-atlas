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

if __name__ == '__main__':
    # path = "data/official/train.csv"
    # annotations = get_annotations(path)
    # print(annotations)

    csv0 = "output/resnet50-512-bce-random-drop0.5-th0.1-bs16-lr0.01-ep15_25.csv"
    csv1 = "output/resnet50-512-official_hpav18-bce-weighted-drop0.5-th0.1-bs16-lr0.005-ep5_15.csv"
    out = "output/0_intersection_1.csv"
    make_union(csv0, csv1, out, "i")
