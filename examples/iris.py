from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data
y = iris.target.astype(np.int32)

model = RandomForestClassifier(max_depth=2, n_estimators=1, random_state=0, bootstrap=False)
model.fit(X, y)

params = {
    'pmml_version': '4.2',
    'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target_values': ['setosa', 'virginica', 'versicolor'],
    'target_name': 'specie',
    'copyright': 'Václav Čadek',
    'description': 'Simple Keras model for Iris dataset.',
    'model_name': 'Iris Model'
}

sklearn2pmml(estimator=model, file='iris.pmml', **params)
export_graphviz(model.estimators_[0], out_file='tree.dot', node_ids=True)
