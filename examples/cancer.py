from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X = cancer.data.astype(np.float32)
y = cancer.target.astype(np.int32)
std = StandardScaler()
X_scaled = std.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

params = {
    'pmml_version': '4.2',
    'feature_names': cancer.feature_names,
    'target_values': cancer.target_names,
    'target_name': 'tumor_type',
    'copyright': 'Václav Čadek',
    'description': 'Simple Logistic Regression model.',
    'model_name': 'Breast Cancer Model'
}

sklearn2pmml(estimator=model, transformer=std, file='cancer.pmml', **params)
