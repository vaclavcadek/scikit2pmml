try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import unittest
from lxml import etree
from scikit2pmml import scikit2pmml
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


class SchemaValidationTestCase:

    @staticmethod
    def _validate_against_schema(schema, pmml):
        with open(schema, 'r') as f:
            schema_root = etree.XML(f.read())
        schema = etree.XMLSchema(schema_root)
        parser = etree.XMLParser(schema=schema)
        etree.fromstring(ET.tostring(pmml.getroot(), encoding='utf-8', method='xml'), parser)

    def test_schema_4_1(self):
        tree = scikit2pmml(estimator=self.model, pmml_version='4.1')
        self._validate_against_schema('xsd/pmml-4-1.xsd', tree)

    def test_schema_4_2(self):
        tree = scikit2pmml(estimator=self.model, pmml_version='4.2')
        self._validate_against_schema('xsd/pmml-4-2.xsd', tree)

    def test_schema_4_3(self):
        tree = scikit2pmml(estimator=self.model, pmml_version='4.3')
        self._validate_against_schema('xsd/pmml-4-3.xsd', tree)


class DecisionTreeClassifierTestCase(SchemaValidationTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = DecisionTreeClassifier()
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        self.model.fit(X, y)


class ExtraTreeClassifierTestCase(SchemaValidationTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = ExtraTreeClassifier()
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        self.model.fit(X, y)


class LinearRegressionTestCase(SchemaValidationTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = LinearRegression()
        boston = load_boston()
        X = boston.data.astype(np.float32)
        y = boston.target.astype(np.float32)
        self.model.fit(X, y)


class LogisticRegressionClassifierTestCase(SchemaValidationTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = LogisticRegression()
        cancer = load_breast_cancer()
        X = cancer.data.astype(np.float32)
        y = cancer.target.astype(np.int32)
        self.model.fit(X, y)


class RandomForestClassifierTestCase(SchemaValidationTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = RandomForestClassifier()
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        self.model.fit(X, y)


class ExtraTreesClassifierTestCase(SchemaValidationTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = ExtraTreesClassifier()
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        self.model.fit(X, y)
