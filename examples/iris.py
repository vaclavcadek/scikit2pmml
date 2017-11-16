from scikit2pmml import scikit2pmml
from sklearn.datasets import load_iris
import numpy as np
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int32)

model = RandomForestClassifier(max_depth=2, n_estimators=10, random_state=0)
model.fit(X, y)

params = {
    'pmml_version': '4.2',
    'feature_names': iris.feature_names,
    'target_values': iris.target_names,
    'target_name': 'specie',
    'copyright': 'Václav Čadek',
    'description': 'Simple Iris RF model.',
    'model_name': 'Iris Model'
}

scikit2pmml(estimator=model, file='iris.pmml', **params)
