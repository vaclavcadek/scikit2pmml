import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree
from scikit2pmml import scikit2pmml


class GenericModelMixin:

    @property
    def is_classification(self):
        return self.dataset.target.dtype != float

    def prepare_model(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.model.fit(dataset.data, dataset.target)
        self.num_inputs = dataset.data.shape[1]
        self.num_outputs = len(dataset.target_names) if self.is_classification else 1
        self.features = ['x{}'.format(i) for i in range(self.num_inputs)]
        self.class_names = ['y{}'.format(i) for i in range(self.num_outputs)]

    def test_data_dict(self):
        pmml = scikit2pmml(self.model)
        continuous_fields = pmml.findall("DataDictionary/DataField/[@optype='continuous']")
        categorical_field = pmml.findall("DataDictionary/DataField/[@optype='categorical']")
        if self.is_classification:
            self.assertEqual(len(continuous_fields), self.num_inputs, 'Correct number of continuous fields.')
            self.assertEqual(len(categorical_field), 1, 'Exactly one categorical field in data dictionary.')
            categorical_name = categorical_field[0].attrib.get('name', None)
            self.assertEqual(categorical_name, 'class', 'Correct target variable name.')
            output_values = categorical_field[0].findall('Value')
            self.assertEqual(len(output_values), self.num_outputs, 'Correct number of output values.')
            self.assertListEqual([ov.attrib['value'] for ov in output_values], self.class_names)
            self.assertListEqual([ov.attrib['name'] for ov in continuous_fields], self.features)
        else:
            self.assertEqual(len(continuous_fields), self.num_inputs + 1, 'Correct number of continuous fields.')

    def test_model(self):
        raise NotImplementedError()


class SchemaValidationMixin:

    @staticmethod
    def _validate_against_schema(schema, pmml):
        with open(schema, 'r') as f:
            schema_root = etree.XML(f.read())
        schema = etree.XMLSchema(schema_root)
        parser = etree.XMLParser(schema=schema)
        etree.fromstring(ET.tostring(pmml.getroot(), encoding='utf-8', method='xml'), parser)

    def test_schema_4_1(self):
        tree = scikit2pmml(estimator=self.model, pmml_version='4.1')
        self._validate_against_schema('{}/xsd/pmml-4-1.xsd'.format(os.path.dirname(os.path.abspath(__file__))), tree)

    def test_schema_4_2(self):
        tree = scikit2pmml(estimator=self.model, pmml_version='4.2')
        self._validate_against_schema('{}/xsd/pmml-4-2.xsd'.format(os.path.dirname(os.path.abspath(__file__))), tree)

    def test_schema_4_3(self):
        tree = scikit2pmml(estimator=self.model, pmml_version='4.3')
        self._validate_against_schema('{}/xsd/pmml-4-3.xsd'.format(os.path.dirname(os.path.abspath(__file__))), tree)
