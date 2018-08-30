from sklearn.tree import DecisionTreeClassifier
from sklearn import neural_network, tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from pandas import read_excel, DataFrame
from numpy import array2string, inf, ravel
import numpy as np
import glob
from itertools import repeat
import pickle

extension = "xlsx"
files = [i for i in glob.glob("*.{}".format(extension))]

c_int = 0

no_of_repeats = len(files)
filelist = [name for name in files if name.endswith(".xlsx")]
for _ in repeat(None, no_of_repeats):
    cfile = filelist[c_int]

    print(cfile, "loading")
    new_file = read_excel(cfile)
    data = new_file

    # Target Variable should be placed on row 2 under class variable in opened file
    target = (DataFrame(data) == 'Target Variable').idxmax(axis=1)[0]
    targname = (DataFrame(data) == 'Target Variable').idxmax(axis=1)[0]
    data = DataFrame(data).astype(dtype='int64', errors='ignore')
    data = DataFrame(data)[1:]
    target = data.filter([target])
    target = DataFrame(target)
    data = DataFrame(data).drop(columns=[targname], axis=1)
    target = target.astype(dtype='int64', errors='ignore')
    varray = list(data)

    # Changes NaN values to 0, only use for testing
    # data = data.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5)

    writedat = open("classdata" + cfile + ".txt", "w")

    tclassify = DecisionTreeClassifier(max_depth=5)
    adaclassify = AdaBoostClassifier(learning_rate=0.25, base_estimator=GradientBoostingClassifier())
    nnclassify = neural_network.MLPClassifier(activation="relu", solver="adam", hidden_layer_sizes=400, alpha=0.08, max_iter=600)

    c_tree = tclassify.fit(X_train, y_train)
    tree_pred = tclassify.predict(X_test)
    tmatrix = (confusion_matrix(y_test, tree_pred))
    t_text = np.array2string(tmatrix, max_line_width=inf)

    tree.export_graphviz(c_tree, out_file='tree(' + cfile + ").dot", feature_names=varray)

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
    c_int = c_int +1

    tfilename = "Classification Tree(" + cfile + ") Attributes" + ".txt"
    twrite = open(tfilename, "w")

    tdata = [tclassify.classes_, tclassify.feature_importances_, tclassify.max_features, tclassify.n_classes_, tclassify.n_features_, tclassify.n_outputs_, tclassify.tree_]
    twrite.writelines(str(tdata))

    adfilename = "Adaboost(" + cfile + ") Attributes" + ".txt"
    adwrite = open(adfilename, "w")

    addata = [adaclassify.estimators_, adaclassify.classes_, adaclassify.n_classes_, adaclassify.estimator_weights_, adaclassify.estimator_errors_, adaclassify.feature_importances_]
    adwrite.writelines(str(addata))

    nnfilename = "Neural Network(" + cfile + ") Attributes" + ".txt"
    nnwrite = open(nnfilename, "w")
    data_string = pickle.dumps(nnclassify)
    np.set_printoptions(threshold=9999999999999999999999999999)

    nndata = [nnclassify.classes_, nnclassify.loss, nnclassify.coefs_, nnclassify.intercepts_, nnclassify.n_iter_, nnclassify.n_layers_, nnclassify.n_outputs_, nnclassify.out_activation_ ]
    nnwrite.writelines(str(nndata))

    print("File ", cfile, "has been saved" )

