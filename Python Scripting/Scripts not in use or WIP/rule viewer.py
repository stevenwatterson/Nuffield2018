from Orange import classification
from Orange.data import Table
import Orange
import glob
import sys
import sklearn
import numpy as np
import AnyQt
from Orange.widgets.visualize import owtreeviewer
from os import system
import numpy as np
import os
import tempfile


extension = 'csv'
files = [i for i in glob.glob('*.{}'.format(extension))]


fileList = [name for name in files if name.endswith(".csv")]

cnt = 0
for cnt, fileName in enumerate(fileList, 0):
    sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

choice = int(input("Select file to load[0-%s]: " % cnt))


data: Table = Table(files[choice])

neural = classification.NNClassificationLearner(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, preprocessors=None)
forest = classification.RandomForestLearner(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, class_weight=None, preprocessors=None)
tree = classification.SklTreeLearner(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, max_leaf_nodes=None, preprocessors=None)

learners = [neural, forest, tree]


print(" "*9 + " ".join("%-4s" % learner.name for learner in learners))
res = Orange.evaluation.CrossValidation(data, learners, k=5)
print("Accuracy %s" % " ".join("%.2f" % s for s in Orange.evaluation.CA(res)))
print("AUC      %s" % " ".join("%.2f" % s for s in Orange.evaluation.AUC(res)))

import sys
from AnyQt.QtWidgets import QApplication, QGraphicsScene, QGraphicsRectItem
from Orange.modelling.tree import TreeLearner

a = QApplication(sys.argv)
ow = owtreeviewer.OWTreeGraph()
clf = TreeLearner()(data)
clf.instances = data

from Orange.widgets.tests import base
from Orange.widgets import io
ow.ctree(clf)
ow.show()
ow.raise_()
a.exec_()
ow.saveSettings()

import matplotlib
Orange.widgets.visualize.owtreeviewer.OWTreeGraph().save_graph(clf, fil)