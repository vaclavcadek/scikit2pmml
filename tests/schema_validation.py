try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import unittest
import os
from lxml import etree
from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


class SchemaValidationTestCase(unittest.TestCase):

    def test_datasets(self):
        scenarios = [
            (RandomForestClassifier(max_depth=4, n_estimators=10, random_state=0), load_iris),
            (LogisticRegression(), load_breast_cancer),
            (LinearRegression(), load_boston)
        ]
        for estimator, load_dataset in scenarios:
            dataset = load_dataset()
            X = dataset.data.astype(np.float32)
            y = dataset.target.astype(np.int32)
            estimator.fit(X, y)

            for schema_file in os.listdir('xsd'):
                version = '.'.join(schema_file.split('.')[0].split('-')[1:])
                tree = sklearn2pmml(estimator=estimator, pmml_version=version)
                with open('xsd/{}'.format(schema_file), 'r') as f:
                    schema_root = etree.XML(f.read())
                schema = etree.XMLSchema(schema_root)
                parser = etree.XMLParser(schema=schema)
                etree.fromstring(ET.tostring(tree.getroot(), encoding='utf-8', method='xml'), parser)
