import pandas as pd


def get_annotations(train_csv):
    df = pd.read_csv(train_csv, index_col=0)
    columns = df.columns
    annotations = df.to_dict(orient="index")
    annotations = {k: [int(x) for x in sorted(v[columns[-1]].split())] for k, v in annotations.items()}
    return annotations


def make_union(csv0, csv1, out):
    df0 = pd.read_csv(csv0, index_col=0)
    df1 = pd.read_csv(csv1, index_col=0)

    ids = df0.index.tolist()
    print(len(ids))

    labels0 = df0.to_dict(orient="index")
    labels0 = {k: [x for x in sorted(str(v[df0.columns[-1]]).split())] for k, v in labels0.items()}

    labels1 = df1.to_dict(orient="index")
    labels1 = {k: [x for x in sorted(str(v[df1.columns[-1]]).split())] for k, v in labels1.items()}

    labels = [' '.join(sorted(list(set(labels0[id_]).union(set(labels1[id_]))))) for id_ in ids]
    df = pd.DataFrame({'Id':ids,'Predicted':labels})
    print(df)
    df.to_csv(out, header=True, index=False)

if __name__ == '__main__':
    # path = "data/official/train.csv"
    # annotations = get_annotations(path)
    # print(annotations)

    csv0 = "output/resnet50_224_ep5_15.csv"
    csv1 = "output/resnet50_224_ep5_5.csv"
    out = "output/resnet50_224_ep5_5_intersection_15.csv"
    make_union(csv0, csv1, out)
