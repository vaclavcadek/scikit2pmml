import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree
from sklearn2pmml import sklearn2pmml


class GenericModelMixin:

    def prepare_model(self, model, dataset, X_type=np.float32, y_type=np.int32):
        self.model = model
        self.dataset = dataset
        self.model.fit(dataset.data.astype(X_type),  dataset.target.astype(y_type))
        self.num_inputs = len(dataset.feature_names)
        self.num_outputs = len(dataset.target_names)
        self.features = ['x{}'.format(i) for i in range(self.num_inputs)]
        self.class_names = ['y{}'.format(i) for i in range(self.num_outputs)]

    def test_data_dict(self):
        pmml = sklearn2pmml(self.model)
        continuous_fields = pmml.findall("DataDictionary/DataField/[@optype='continuous']")
        categorical_field = pmml.findall("DataDictionary/DataField/[@optype='categorical']")
        self.assertEqual(len(continuous_fields), self.num_inputs, 'Correct number of continuous fields.')
        self.assertEqual(len(categorical_field), 1, 'Exactly one categorical field in data dictionary.')
        categorical_name = categorical_field[0].attrib.get('name', None)
        self.assertEqual(categorical_name, 'class', 'Correct target variable name.')
        output_values = categorical_field[0].findall('Value')
        self.assertEqual(len(output_values), self.num_outputs, 'Correct number of output values.')
        self.assertListEqual([ov.attrib['value'] for ov in output_values], self.class_names)
        self.assertListEqual([ov.attrib['name'] for ov in continuous_fields], self.features)

    # def test_mining_schema(self):
    #     target_field = self.pmml.findall("MiningSchema/MiningField/[@usageType='target']")
    #     active_fields = self.pmml.findall("MiningSchema/MiningField/[@usageType='active']")
    #     self.assertEqual(len(active_fields), self.num_inputs, 'Correct number of active fields.')
    #     self.assertEqual(len(target_field), 1, 'Exactly one target field in mining schema.')
    #     target_name = target_field[0].attrib.get('name', None)
    #     self.assertEqual(target_name, 'class', 'Correct target field name.')
    #     self.assertListEqual(
    #         [ov.attrib['name'] for ov in active_fields],
    #         self.features
    #     )
    #
    # def test_output(self):
    #     output_fields = self.pmml.findall("Output/OutputField/[@feature='probability']")
    #     self.assertEqual(len(output_fields), self.num_outputs, 'Correct number of output fields.')
    #     self.assertListEqual(
    #         [of.attrib['name'] for of in output_fields],
    #         ['probability_{}'.format(v) for v in self.class_names]
    #     )

    def test_model(self):
        raise NotImplementedError()
        # trees = self.pmml.findall('MiningModel/Segmentation/Segment/TreeModel')
        # self.assertEqual(len(trees), self.num_trees, 'Correct number of trees.')


class SchemaValidationMixin:

    @staticmethod
    def _validate_against_schema(schema, pmml):
        with open(schema, 'r') as f:
            schema_root = etree.XML(f.read())
        schema = etree.XMLSchema(schema_root)
        parser = etree.XMLParser(schema=schema)
        etree.fromstring(ET.tostring(pmml.getroot(), encoding='utf-8', method='xml'), parser)

    def test_schema_4_1(self):
        tree = sklearn2pmml(estimator=self.model, pmml_version='4.1')
        self._validate_against_schema('xsd/pmml-4-1.xsd', tree)

    def test_schema_4_2(self):
        tree = sklearn2pmml(estimator=self.model, pmml_version='4.2')
        self._validate_against_schema('xsd/pmml-4-2.xsd', tree)

    def test_schema_4_3(self):
        tree = sklearn2pmml(estimator=self.model, pmml_version='4.3')
        self._validate_against_schema('xsd/pmml-4-3.xsd', tree)