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
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob
import sys
import os

extension = "xlsx"
files = [i for i in glob.glob("*.{}".format(extension))]


filelist = [name for name in files if name.endswith(".xlsx")]
for name in filelist:
    c_int = 0
    cfile = filelist[c_int]



    print(cfile, "loading")
    new_file = read_excel(cfile)
    data = new_file

    data = data.fillna(0)

    target = (DataFrame(data) == 'Target Variable').idxmax(axis=1)[0]
    targname = (DataFrame(data) == 'Target Variable').idxmax(axis=1)[0]
    print (targname)
    data = DataFrame(data).astype(dtype='int64', errors='ignore')
    data = DataFrame(data)[1:]
    print(data)
    target = data.filter([target])
    target = DataFrame(target)
    print(target)
    data = DataFrame(data).drop(columns=[targname], axis=1)
    target = target.astype(dtype='int64', errors='ignore')
    varray = list(data)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    dlist = []
    dlist.append('classdata(')
    dlist.append(cfile)
    dlist.append(').dot')

    dname = ''.join(dlist)
    writedat = open("classdata" + cfile + ".txt", "w")


    tclassify = DecisionTreeClassifier()
    adaclassify = AdaBoostClassifier()
    nnclassify = neural_network.MLPClassifier(activation="relu", solver="sgd", hidden_layer_sizes=400, alpha=0.05, max_iter=400)

    c_tree = tclassify.fit(X_train, y_train)
    tree_pred = tclassify.predict(X_test)
    tmatrix = (confusion_matrix(y_test, tree_pred))
    t_text = np.array2string(tmatrix, max_line_width=inf)

    tlist = []
    tlist.append('tree')
    tlist.append(cfile)
    tlist.append('.dot')

    tname = ''.join(tlist)
    tree.export_graphviz(c_tree, out_file='tree' + cfile + ".dot", feature_names=varray)

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
    c_int + 1

print("Script has finished.")
