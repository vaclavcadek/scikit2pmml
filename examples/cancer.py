from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target.astype(np.int32)

model = RandomForestClassifier(max_depth=6, n_estimators=100, random_state=0)
model.fit(X, y)

params = {
    'pmml_version': '4.2',
    'feature_names': cancer.feature_names,
    'target_values': cancer.target_names,
    'target_name': 'tumor_type',
    'copyright': 'Václav Čadek',
    'description': cancer.DESCR,
    'model_name': 'Breast Cancer Model'
}

sklearn2pmml(estimator=model, file='cancer.pmml', **params)
