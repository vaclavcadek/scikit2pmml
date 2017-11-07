import unittest

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from tests.models.generic import GenericModelMixin
from tests.models.generic import SchemaValidationMixin


class DecisionTreeClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(DecisionTreeClassifier(), load_iris())

    def test_model(self):
        pass


class ExtraTreeClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(ExtraTreeClassifier(), load_iris())

    def test_model(self):
        pass
