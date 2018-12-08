import pandas as pd


def get_annotations(train_csv):
    df = pd.read_csv(train_csv, index_col=0)
    columns = df.columns
    annotations = df.to_dict(orient="index")
    annotations = {k: [int(x) for x in sorted(v[columns[-1]].split())] for k, v in annotations.items()}
    return annotations


if __name__ == '__main__':
    path = "data/official/train.csv"
    annotations = get_annotations(path)
    print(annotations)
