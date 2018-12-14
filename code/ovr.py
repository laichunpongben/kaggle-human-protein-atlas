# Adopted from https://www.kaggle.com/philschmidt/multilabel-classification-rainforest-eda

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns
from glob import glob
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score
import cv2
import warnings

sample = pd.read_csv('data/official/sample_submission.csv')
df = pd.read_csv('data/official/train.csv')

n_samples = 31072
rescaled_dim = 224

df['split_tags'] = df['Target'].map(lambda row: row.split(" "))
lb = MultiLabelBinarizer()
y = lb.fit_transform(df['split_tags'])
y = y[:n_samples]
X = np.squeeze(np.array([cv2.resize(plt.imread('data/rgb/train/{}.png'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in df.head(n_samples)['Id'].values]))
X = MinMaxScaler().fit_transform(X)

print(X.shape, y.shape, lb.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = OneVsRestClassifier(LogisticRegression(C=10, penalty='l2'))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    clf.fit(X_train, y_train)

score = fbeta_score(y_test, clf.predict(X_test), beta=2, average=None)
avg_sample_score = fbeta_score(y_test, clf.predict(X_test), beta=2, average='samples')
print('Average F2 test score {}'.format(avg_sample_score))
print('F2 test scores per tag:')

X_sub = np.squeeze(np.array([cv2.resize(plt.imread('data/rgb/test/{}.png'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in sample['Id'].values]))
X_sub = MinMaxScaler().fit_transform(X_sub)

y_sub = clf.predict(X_sub)
all_test_tags = []
for index in range(y_sub.shape[0]):
    all_test_tags.append(' '.join(list(lb.classes_[np.where(y_sub[index, :] == 1)[0]])))

test_imgs = [plt.imread('data/rgb/test/{}.png'.format(name)) for name in sample.head(20)['Id'].values]
sample['Predicted'] = all_test_tags
sample.to_csv('ovr_f2_{}.csv'.format(avg_sample_score), index=False)
