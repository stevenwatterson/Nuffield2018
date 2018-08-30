from pandas as pd
from sklearn import tree

data = pd.read_csv('cvdorange.csv')
classifier = tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

tree.export_graphviz(decision_tree, out_file=”tree.dot”, max_depth=None, feature_names=None, class_names=None, label=’all’, filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3)