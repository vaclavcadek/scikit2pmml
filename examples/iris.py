from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

model = RandomForestClassifier(max_depth=2)
model.fit(X_train, y_train)

params = {
    'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target_values': ['setosa', 'virginica', 'versicolor'],
    'target_name': 'specie',
    'copyright': 'Václav Čadek',
    'description': 'Simple Keras model for Iris dataset.',
    'model_name': 'Iris Model'
}

sklearn2pmml(estimator=model, file='iris.pmml', **params)
