try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from sklearn2pmml.models.tree import TreeModel
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


SUPPORTED_MODELS = frozenset([
    DecisionTreeClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
])
SUPPORTED_TRANSFORMERS = frozenset([StandardScaler, MinMaxScaler])
SUPPORTED_NS = {
    '4.1': 'http://www.dmg.org/PMML-4_1',
    '4.2': 'http://www.dmg.org/PMML-4_2',
    '4.2.1': 'http://www.dmg.org/PMML-4_2-',
    '4.3': 'http://www.dmg.org/PMML-4_3'
}


class PMMLDocument:

    def __init__(self, estimator, transformer, **kwargs):
        self.estimator = estimator
        self.transformer = transformer
        self.feature_names = kwargs.get('feature_names', [])
        self.target_name = kwargs.get('target_name', 'class')
        self.target_values = kwargs.get('target_values', [])
        self.version = kwargs.get('version', '4.2')
        self.model_name = kwargs.get('model_name', None)
        self.description = kwargs.get('description', None)
        self.copyright = kwargs.get('copyright', None)
        self.root = None

    @staticmethod
    def _get_model(estimator):
        if type(estimator) == DecisionTreeClassifier:
            return TreeModel(estimator)
        if type(estimator) in [RandomForestClassifier, ExtraTreesClassifier]:
            return [TreeModel(tree) for tree in estimator.estimators_]

    def _validate_inputs(self):
        logger.info('[x] Performing model validation.')
        if not type(self.estimator) in SUPPORTED_MODELS:
            raise TypeError("Provided model is not supported.")
        if not self.estimator.fit:
            raise TypeError("Provide a fitted model.")
        if self.transformer is not None and not type(self.transformer) in SUPPORTED_TRANSFORMERS:
            raise TypeError("Provided transformer is not supported.")
        if self.estimator.n_features_ != len(self.feature_names):
            logger.warning('[!] Input shape does not match provided feature names - using generic names instead.')
            self.feature_names = ['x{}'.format(i) for i in range(self.estimator.n_features_)]
        if self.estimator.n_classes_ != len(self.target_values):
            logger.warning('[!] Output shape does not match provided target values - using generic names instead.')
            self.target_values = ['y{}'.format(i) for i in range(self.estimator.n_classes_)]
        logger.info('[x] Model validation successful.')

    def generate_document(self, file):
        self._validate_inputs()
        self.root = ET.Element('PMML')
        self.root.set('version', self.version)
        self.root.set('xmlns', SUPPORTED_NS.get(self.version, 'http://www.dmg.org/PMML-4_2'))
        self._generate_header()
        self._generate_data_dictionary()
        self._generate_model()
        tree = ET.ElementTree(self.root)
        logger.info('[x] Generation of PMML successful.')
        if file:
            tree.write(file, encoding='utf-8', xml_declaration=True)
        return tree

    def _generate_header(self):
        header = ET.SubElement(self.root, 'Header')
        if self.copyright:
            header.set('copyright', self.copyright)
        if self.description:
            header.set('description', self.description)
        timestamp = ET.SubElement(header, 'Timestamp')
        timestamp.text = str(datetime.now())
        return header

    def _generate_data_dictionary(self):
        data_dict = ET.SubElement(self.root, 'DataDictionary')
        data_field = ET.SubElement(data_dict, 'DataField')
        data_field.set('name', self.target_name)
        data_field.set('dataType', 'string')
        data_field.set('optype', 'categorical')
        logger.info('[x] Generating Data Dictionary:')
        for t in self.target_values:
            value = ET.SubElement(data_field, 'Value')
            value.set('value', t)
        for f in self.feature_names:
            data_field = ET.SubElement(data_dict, 'DataField')
            data_field.set('name', f)
            data_field.set('dataType', 'double')
            data_field.set('optype', 'continuous')
            logger.info('\t[-] {}...OK!'.format(f))
        return data_dict

    def _generate_model(self):
        model = self._get_model(self.estimator)
        if type(model) == list:
            mining_model = ET.SubElement(self.root, 'MiningModel')
            mining_model.set('functionName', 'classification')
            if self.model_name:
                mining_model.set('modelName', self.model_name)
            self._generate_mining_schema(mining_model)
            self._generate_output(mining_model)
            segmentation = ET.SubElement(mining_model, 'Segmentation')
            segmentation.set('multipleModelMethod', 'average')
            for i, m in enumerate(model):
                segment = ET.SubElement(segmentation, 'Segment')
                segment.set('id', str(i))
                ET.SubElement(segment, 'True')
                model_element = m.serialize(segment, self.feature_names, self.target_values)
                self._generate_mining_schema(model_element)
            return mining_model
        else:
            model_element = model.serialize(self.root, self.feature_names, self.target_values)
            self._generate_mining_schema(model_element)
            return model_element

    def _generate_mining_schema(self, parent):
        mining_schema = ET.SubElement(parent, 'MiningSchema')
        mining_field = ET.SubElement(mining_schema, 'MiningField')
        if self.target_name:
            mining_field.set('name', self.target_name)
            mining_field.set('usageType', 'target')
        for f in self.feature_names:
            mining_field = ET.SubElement(mining_schema, 'MiningField')
            mining_field.set('name', f)
            mining_field.set('usageType', 'active')
        return mining_schema

    def _generate_output(self, parent):
        output = ET.SubElement(parent, 'Output')
        for t in self.target_values:
            output_field = ET.SubElement(output, 'OutputField')
            output_field.set('name', 'probability_{}'.format(t))
            output_field.set('feature', 'probability')
            output_field.set('value', t)
        return output


def sklearn2pmml(estimator, transformer=None, file=None, **kwargs):
    """
    Exports sklearn model as PMML.

    :param estimator: sklearn model to be exported as PMML (for supported models - see bellow).
    :param transformer: if provided then scaling is applied to data fields.
    :param file: name of the file where the PMML will be exported.
    :param kwargs: set of params that affects PMML metadata - see documentation for details.
    :return: XML element tree
    """

    document = PMMLDocument(estimator, transformer, **kwargs)
    return document.generate_document(file)


