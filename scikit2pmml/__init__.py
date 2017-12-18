try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from scikit2pmml.models.tree import TreeModel
from scikit2pmml.models.ensemble import Segmentation
from scikit2pmml.models.regression import RegressionModel
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

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
        self.version = kwargs.get('pmml_version', '4.2')
        self.model_name = kwargs.get('model_name', None)
        self.description = kwargs.get('description', None)
        self.copyright = kwargs.get('copyright', None)
        self.root = None
        self.serializer = self._get_serializer(estimator)

    def _get_serializer(self, estimator):
        if type(estimator) == LinearRegression:
            return RegressionModel(estimator, self, 'regression', RegressionModel.LINEAR_REGRESSION)
        if type(estimator) == LogisticRegression:
            return RegressionModel(estimator, self, 'classification', RegressionModel.LOGISTIC_REGRESSION)
        if type(estimator) in [DecisionTreeClassifier, ExtraTreeClassifier]:
            return TreeModel(estimator, self, 'classification')
        if type(estimator) in [RandomForestClassifier, ExtraTreesClassifier]:
            return Segmentation(estimator, self, 'classification')
        raise TypeError("Provided model is not supported.")

    def _validate_inputs(self):
        logger.info('[x] Performing model validation.')
        if not self.estimator.fit:
            raise TypeError("Provide a fitted model.")
        if self.transformer is not None and not type(self.transformer) in SUPPORTED_TRANSFORMERS:
            raise TypeError("Provided transformer is not supported.")
        if self.serializer.n_features != len(self.feature_names):
            logger.warning('[!] Input shape does not match provided feature names - using generic names instead.')
            self.feature_names = ['x{}'.format(i) for i in range(self.serializer.n_features)]
        if self.serializer.function_name == 'classification' and self.serializer.n_classes != len(self.target_values):
            logger.warning('[!] Output shape does not match provided target values - using generic names instead.')
            self.target_values = ['y{}'.format(i) for i in range(self.serializer.n_classes)]
        logger.info('[x] Model validation successful.')

    @property
    def document(self):
        self._validate_inputs()
        self.root = ET.Element('PMML')
        self.root.set('version', self.version)
        self.root.set('xmlns', SUPPORTED_NS.get(self.version, 'http://www.dmg.org/PMML-4_2'))
        self.root.append(self.header)
        self.root.append(self.data_dictionary)
        self.root.append(self.model)
        tree = ET.ElementTree(self.root)
        logger.info('[x] Generation of PMML successful.')
        return tree

    @property
    def header(self):
        header = ET.Element('Header')
        if self.copyright:
            header.set('copyright', self.copyright)
        if self.description:
            header.set('description', self.description)
        timestamp = ET.SubElement(header, 'Timestamp')
        timestamp.text = str(datetime.now())
        return header

    @property
    def data_dictionary(self):
        data_dict = ET.Element('DataDictionary')
        data_field = ET.SubElement(data_dict, 'DataField')
        data_field.set('name', self.target_name)
        data_field.set('dataType', 'string' if self.serializer.function_name == 'classification' else 'double')
        data_field.set('optype', 'categorical' if self.serializer.function_name == 'classification' else 'continuous')
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

    @property
    def model(self):
        return self.serializer.model


def scikit2pmml(estimator, transformer=None, file=None, **kwargs):
    """
    Exports sklearn model as PMML.

    :param estimator: sklearn model to be exported as PMML (for supported models - see bellow).
    :param transformer: if provided then scaling is applied to data fields.
    :param file: name of the file where the PMML will be exported.
    :param kwargs: set of params that affects PMML metadata - see documentation for details.
    :return: XML element tree
    """

    pmml = PMMLDocument(estimator, transformer, **kwargs)
    tree = pmml.document
    if file:
        tree.write(file, encoding='utf-8', xml_declaration=True)
    return tree


