import os
import pandas as pd
import numpy as np
from sklearn import svm
from config import TEST_PATH
from code.py_scripts.color_histogram_helper import get_ids, get_intersections

train = pd.read_csv("input/train_flattened.csv")
x_columns = ["RG", "RB", "RY", "GB", "GY", "BY"]
x_train = train[x_columns]
# print(x_train.head(1))

num_class = 28
y_trains = [train[str(x)] for x in range(num_class)]
clfs = [svm.SVC() for x in range(num_class)]
for clf, y_train in zip(clfs, y_trains):
    clf.fit(x_train, y_train)
print(x_train)

test_ids = get_ids(TEST_PATH)
x_test = pd.DataFrame(index=test_ids)

# x_test = x_test.head(100)

x_columns = ["RG", "RB", "RY", "GB", "GY", "BY"]
x_test['Intersections'] = x_test.index.map(get_intersections)
x_test[x_columns] = pd.DataFrame(x_test.Intersections.values.tolist(), index=x_test.index)
x_test = x_test.drop(['Intersections'], axis=1)

y_test = pd.DataFrame(index=x_test.index)
print(y_test)
for i, clf in enumerate(clfs):
    func = lambda row: clf.predict(np.reshape(row[x_columns].values, (1, -1)))[0]
    y_test[str(i)] = x_test.apply(func, axis=1)

print(y_test)
out_flattened_path = os.path.join("output", "color_histogram_svm_flattened_0.csv")
y_test.to_csv(out_flattened_path,
              encoding="utf-8")
              # header=["Id"]+[str(x) for x in range(num_class)])
