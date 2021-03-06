from sklearn.tree import DecisionTreeClassifier
from sklearn import neural_network, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pandas import read_excel, DataFrame
from numpy import array2string, inf, ravel
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import sys
import os

extension = "xlsx"
files = [i for i in glob.glob("*.{}".format(extension))]


fileList = [name for name in files if name.endswith(".xlsx")]

cnt = 0
for cnt, fileName in enumerate(fileList, 0):
    sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

choice = int(input("Select file to load[0-%s]: " % cnt))

print(files[choice], "loading")
new_file = read_excel(files[choice])
data = new_file

target = DataFrame(data)

targ = str(input("Type in target class name as it is on the spreadsheet(case sensitive): "))

while set([targ]).issubset(DataFrame(data)) is False:
    print(targ, "variable has not been found. Please try again")
    time.sleep(1)
    targ = str(input("Type in target class name as it is on the spreadsheet(case sensitive): "))

else:
    print(targ, "is now selected as the target variable")

data = data.fillna(0)


DataFrame(data).set_index([targ])
target = target[[targ]].copy()
data = DataFrame(data).drop([targ], axis=1)
varray = list(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

writedat = open("classdata.txt", "w")


tclassify = DecisionTreeClassifier()
adaclassify = AdaBoostClassifier()
nnclassify = neural_network.MLPClassifier(activation="relu", solver="sgd", hidden_layer_sizes=400, alpha=0.05, max_iter=400)

c_tree = tclassify.fit(X_train, y_train)
tree_pred = tclassify.predict(X_test)
tmatrix = (confusion_matrix(y_test, tree_pred))
t_text = np.array2string(tmatrix, max_line_width=inf)

tree.export_graphviz(c_tree, out_file='tree.dot', feature_names=varray)

ada_pipeline = make_pipeline(StandardScaler(), adaclassify)
yravel = ravel(y_train)


c_ada = ada_pipeline.fit(X_train, yravel)
ada_pred = ada_pipeline.predict(X_test)
adamatrix = (confusion_matrix(y_test, tree_pred))
adatext = np.array2string(adamatrix, max_line_width=inf)

nn_pipeline = make_pipeline(StandardScaler(), nnclassify)

nn_pipeline.fit(X_train, yravel)
nn_pipeline.score(X_test, y_test)

nn_pred = nn_pipeline.predict(X_test)
nnmatrix = (confusion_matrix(y_test, nn_pred))
nntext = np.array2string(nnmatrix, max_line_width=inf)


stringdata =["|--Tree Classification--|", '', "|--Confusion Matrix--|", '', t_text, '', "|--Classification Data--|", '', classification_report(y_test, tree_pred), '', "|--Settings used--|", '', str(tclassify), '', "|--AdaBoost Classification--|", '', "|--Confusion Matrix--|", '', adatext, '', "|--Classification Data--|", '',
str(classification_report(y_test, ada_pred)), '', "|--Settings Used--|", str(adaclassify), '', "|--Neural Network Classification--|", '', "|--Confusion Matrix--|", '', nntext, '', "|--Classification Data--|", '', str(classification_report(y_test, nn_pred)), '',
"|--Settings used--|", '', str(nnclassify)]

writedat.writelines(stringdata)

print("Script has finished.")
