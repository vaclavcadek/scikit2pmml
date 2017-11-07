from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from tests.models.generic import GenericModelMixin
from tests.models.generic import SchemaValidationMixin
import numpy as np
import unittest


class LinearRegressionTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(LinearRegression(), load_boston(), np.float32, np.float32)

    def test_model(self):
        pass


class LogisticRegressionClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(LogisticRegression(), load_breast_cancer())

    def test_model(self):
        pass
