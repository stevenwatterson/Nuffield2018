import Orange
from sklearn import tree
import graphviz
import numpy as np
import sklearn
data = Orange.data.Table("cvdprocessed.tab")


data.X = data.X[~np.isnan(data.X)]
data.Y = data.Y[~np.isnan(data.Y)]
clf = tree.DecisionTreeClassifier()

clf = clf.fit(data.X, data.Y)


dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph