import unittest

import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.generic import GenericModelMixin
from tests.generic import SchemaValidationMixin


class LinearRegressionTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(LinearRegression(), load_boston())

    def test_model(self):
        pass


class LogisticRegressionClassifierTestCase(GenericModelMixin, SchemaValidationMixin, unittest.TestCase):

    def setUp(self):
        super().prepare_model(LogisticRegression(), load_breast_cancer())

    def test_model(self):
        pass
