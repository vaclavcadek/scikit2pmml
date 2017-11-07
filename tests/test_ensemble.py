import unittest

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from tests.generic import GenericModelMixin
from tests.generic import SchemaValidationMixin


class RandomForestClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(RandomForestClassifier(), load_iris())

    def test_model(self):
        pass


class ExtraTreesClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(ExtraTreesClassifier(), load_iris())

    def test_model(self):
        pass
