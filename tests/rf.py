import unittest

from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class GenericFieldsTestCase(unittest.TestCase):
    def setUp(self):
        iris = load_iris()

        X = iris.data.astype(np.float64)
        y = iris.target.astype(np.int32)

        model = RandomForestClassifier(max_depth=4)
        model.fit(X, y)

        params = {'copyright': 'Václav Čadek', 'model_name': 'Iris Model'}
        self.model = model
        self.pmml = sklearn2pmml(self.model, **params)
        self.num_trees = len(self.model.estimators_)
        self.num_inputs = model.n_features_
        self.num_outputs = model.n_classes_
        self.features = ['x{}'.format(i) for i in range(self.num_inputs)]
        self.class_names = ['y{}'.format(i) for i in range(self.num_outputs)]

    def test_data_dict(self):
        continuous_fields = self.pmml.findall("DataDictionary/DataField/[@optype='continuous']")
        categorical_field = self.pmml.findall("DataDictionary/DataField/[@optype='categorical']")
        self.assertEquals(len(continuous_fields), self.num_inputs, 'Correct number of continuous fields.')
        self.assertEquals(len(categorical_field), 1, 'Exactly one categorical field in data dictionary.')
        categorical_name = categorical_field[0].attrib.get('name', None)
        self.assertEquals(categorical_name, 'class', 'Correct target variable name.')
        output_values = categorical_field[0].findall('Value')
        self.assertEqual(len(output_values), self.num_outputs, 'Correct number of output values.')
        self.assertListEqual(
            [ov.attrib['value'] for ov in output_values],
            self.class_names
        )
        self.assertListEqual(
            [ov.attrib['name'] for ov in continuous_fields],
            self.features
        )

    def test_mining_schema(self):
        target_field = self.pmml.findall("MiningModel/MiningSchema/MiningField/[@usageType='target']")
        active_fields = self.pmml.findall("MiningModel/MiningSchema/MiningField/[@usageType='active']")
        self.assertEquals(len(active_fields), self.num_inputs, 'Correct number of active fields.')
        self.assertEquals(len(target_field), 1, 'Exactly one target field in mining schema.')
        target_name = target_field[0].attrib.get('name', None)
        self.assertEquals(target_name, 'class', 'Correct target field name.')
        self.assertListEqual(
            [ov.attrib['name'] for ov in active_fields],
            self.features
        )

    def test_output(self):
        output_fields = self.pmml.findall("MiningModel/Output/OutputField/[@feature='probability']")
        self.assertEqual(len(output_fields), self.num_outputs, 'Correct number of output fields.')
        self.assertListEqual(
            [of.attrib['name'] for of in output_fields],
            ['probability_{}'.format(v) for v in self.class_names]
        )

    def test_model(self):
        trees = self.pmml.findall('MiningModel/Segmentation/Segment/TreeModel')
        self.assertEqual(len(trees), self.num_trees, 'Correct number of trees.')

