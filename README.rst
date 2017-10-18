sklearn2pmml
==========

sklearn2pmml is simple exporter for sklearn models (for supported models see bellow) into PMML text format which address
the problems mentioned bellow.

Storing predictive models using binary format (e.g. Pickle) may be dangerous from several perspectives - naming few:

* **binary compatibility**:you update the libraries and may not be able to open the model serialized with older version
* **dangerous code**: when you would use model made by someone else
* **interpretability**: model cannot be easily opened and reviewed by human
* etc.

In addition the PMML is able to persist scaling of the raw input features which helps gradient descent to run smoothly
through optimization space.

Installation
------------

To install sklearn2pmml, simply:

.. code-block:: bash

    $ pip install sklearn2pmml

Example
-------

Example on Iris data - for more examples see the examples folder.

.. code-block:: python

    from sklearn2pmml import sklearn2pmml
    from sklearn.datasets import load_iris
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int32)

    model = RandomForestClassifier(max_depth=2, n_estimators=10, bootstrap=False, random_state=0)
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

    sklearn2pmml(estimator=model, file='iris.pmml', **params)




Params explained
----------------
- **estimator**: Sklearn model to be exported as PMML (for supported models - see bellow).
- **transformer**: if provided (and it's supported - see bellow) then scaling is applied to data fields.
- **file**: name of the file where the PMML will be exported.
- **feature_names**: when provided and have same shape as input layer, then features will have custom names, otherwise generic names (x\ :sub:`0`\,..., x\ :sub:`n-1`\) will be used.
- **target_values**: when provided and have same shape as output layer, then target values will have custom names, otherwise generic names (y\ :sub:`0`\,..., y\ :sub:`n-1`\) will be used.
- **target_name**: when provided then target variable will have custom name, otherwise generic name **class** will be used.
- **copyright**: who is the author of the model.
- **description**: optional parameter that sets *description* within PMML document.
- **model_name**: optional parameter that sets *model_name* within PMML document.

What is supported?
------------------
- Linear Model
    * sklearn.linear_model.LinearRegression
    * sklearn.linear_model.LogisticRegression
- Tree
    * sklearn.tree.DecisionTree
    * sklearn.tree.ExtraTreeClassifier
- Ensemble
    * sklearn.ensemble.RandomForestClassifier
    * sklearn.ensemble.ExtraTreesClassifier
- Scalers
    * sklearn.preprocessing.StandardScaler
    * sklearn.preprocessing.MinMaxScaler

License
-------

This software is licensed under MIT licence.

- https://opensource.org/licenses/MIT