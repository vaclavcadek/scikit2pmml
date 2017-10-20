from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_boston
import numpy as np
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston.data.astype(np.float32)
y = boston.target.astype(np.float32)

model = LinearRegression()
model.fit(X, y)

params = {
    'pmml_version': '4.2',
    'feature_names': boston.feature_names,
    'target_name': 'median_value',
    'copyright': 'Václav Čadek',
    'description': 'Simple Linear Regression model.',
    'model_name': 'Boston pricing model'
}

sklearn2pmml(estimator=model, file='boston.pmml', **params)
