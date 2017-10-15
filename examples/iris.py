from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
import numpy as np
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target.astype(np.int32)

model = RandomForestClassifier(max_depth=4, n_estimators=1, random_state=0, bootstrap=False)
model.fit(X, y)

params = {
    'pmml_version': '4.2',
    'feature_names': iris.feature_names,
    'target_values': iris.target_names,
    'target_name': 'specie',
    'copyright': 'Václav Čadek',
    'description': 'Simple RF model for Iris dataset.',
    'model_name': 'Iris Model'
}

sklearn2pmml(estimator=model, file='iris.pmml', **params)
