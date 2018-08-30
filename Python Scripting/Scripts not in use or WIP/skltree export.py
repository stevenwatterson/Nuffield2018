import sklearn
import Orange
import numpy as np
from sklearn import tree

data = Orange.data.Table("cvdprocessed")

a = data.X
b = data.Y
print(data.domain.attributes)


#ax = np.nan_to_num(a, copy=True)

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(ax, b)

#import graphviz
#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = graphviz.Source(dot_data)
#graph.render("cvd")