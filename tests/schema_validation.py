try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import unittest
from lxml import etree
from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


class SchemaValidationTestCase(unittest.TestCase):

    def test_datasets(self):
        estimators = [
            RandomForestClassifier(max_depth=4, n_estimators=10, random_state=0),
            LogisticRegression(),
            LinearRegression()
        ]
        datasets = [load_iris, load_breast_cancer, load_boston]
        for estimator, load_dataset in zip(estimators, datasets):
            dataset = load_dataset()
            X = dataset.data.astype(np.float32)
            y = dataset.target.astype(np.int32)
            estimator.fit(X, y)

            params = {
                'pmml_version': '4.2'
            }

            tree = sklearn2pmml(estimator=estimator, **params)
            with open('xsd/pmml-4-2.xsd', 'r') as f:
                schema_root = etree.XML(f.read())
            schema = etree.XMLSchema(schema_root)
            xmlparser = etree.XMLParser(schema=schema)
            etree.fromstring(ET.tostring(tree.getroot(), encoding='utf-8', method='xml'), xmlparser)
