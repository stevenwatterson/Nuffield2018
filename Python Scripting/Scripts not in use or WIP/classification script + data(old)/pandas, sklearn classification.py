import pandas as pd
import glob
import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import neural_network, tree
from sklearn.ensemble import AdaBoostClassifier
import sklearn
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
extension = "xlsx"
files = [i for i in glob.glob("*.{}".format(extension))]


fileList = [name for name in files if name.endswith(".xlsx")]

cnt = 0
for cnt, fileName in enumerate(fileList, 0):
    sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

choice = int(input("Select file to load[0-%s]: " % cnt))

print(files[choice], "loading")
new_file = pd.read_excel(files[choice])
data = new_file

target = pd.DataFrame(data)

targ = str(input("Type in target class name as it is on the spreadsheet(case sensitive): "))

while set([targ]).issubset(pd.DataFrame(data)) is False:
    print(targ, "variable has not been found. Please try again")
    time.sleep(1)
    targ = str(input("Type in target class name as it is on the spreadsheet(case sensitive): "))

else:
    print(targ, "is now selected as the target variable")

data = data.fillna(0)


pd.DataFrame(data).set_index([targ])
target = target[[targ]].copy()
data = pd.DataFrame(data).drop([targ], axis=1)
varray = list(data)
print(data)


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.4)

writedat = open("classdata.txt", "w")


tclassify = DecisionTreeClassifier()
adaclassify = AdaBoostClassifier()
nnclassify = neural_network.MLPClassifier(activation="relu", solver="sgd", hidden_layer_sizes=400, alpha=0.05, max_iter=400)

c_tree = tclassify.fit(X_train, y_train)
tree_pred = tclassify.predict(X_test)
tmatrix = (confusion_matrix(y_test, tree_pred))
t_text = np.array2string(tmatrix, max_line_width=np.inf)

tree.export_graphviz(c_tree, out_file='tree.dot', feature_names=varray)

ada_pipeline = make_pipeline(StandardScaler(), adaclassify)
yravel = np.ravel(y_train)


c_ada = ada_pipeline.fit(X_train, yravel)
ada_pred = ada_pipeline.predict(X_test)
adamatrix = (confusion_matrix(y_test, tree_pred))
adatext = np.array2string(adamatrix, max_line_width=np.inf)



nn_pipeline = make_pipeline(StandardScaler(), nnclassify)


nn_pipeline.fit(X_train, yravel)
nn_pipeline.score(X_test, y_test)

nn_pred = nn_pipeline.predict(X_test)
nnmatrix = (confusion_matrix(y_test, nn_pred))
nntext = np.array2string(nnmatrix, max_line_width=np.inf)


stringdata =["---Tree Classification---", '', "---Confusion Matrix---", '', t_text, '', "---Classification Data---", '', classification_report(y_test, tree_pred), '', "---Settings used---", '', str(tclassify), '', "---AdaBoost Classification---", '', "---Confusion Matrix---", '', adatext, '', "---Classification Data---", '',
str(classification_report(y_test, ada_pred)), '', "---Settings Used---", '', "---Neural Network Classification---", '', "---Confusion Matrix---", '', nntext, '', "---Classification Data---", '', str(classification_report(y_test, nn_pred)), '',
"---Settings used---", '', str(nnclassify)]

writedat.writelines(stringdata)
