import unittest

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn2pmml import sklearn2pmml

from tests.generic import GenericModelMixin
from tests.generic import SchemaValidationMixin


class RandomForestClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(RandomForestClassifier(), load_iris())

    def test_model(self):
        pmml = sklearn2pmml(self.model)
        trees = pmml.findall('MiningModel/Segmentation/Segment/TreeModel')
        self.assertEqual(len(trees), len(self.model.estimators_), 'Correct number of trees.')


class ExtraTreesClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(ExtraTreesClassifier(), load_iris())

    def test_model(self):
        pmml = sklearn2pmml(self.model)
        trees = pmml.findall('MiningModel/Segmentation/Segment/TreeModel')
        self.assertEqual(len(trees), len(self.model.estimators_), 'Correct number of trees.')
