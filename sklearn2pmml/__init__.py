try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from sklearn2pmml.models.tree import TreeModel
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


SUPPORTED_MODELS = frozenset([DecisionTreeClassifier, RandomForestClassifier])
SUPPORTED_TRANSFORMERS = frozenset([StandardScaler, MinMaxScaler])
SUPPORTED_NS = {
    '4.1': 'http://www.dmg.org/PMML-4_1',
    '4.2': 'http://www.dmg.org/PMML-4_2',
    '4.2.1': 'http://www.dmg.org/PMML-4_2-',
    '4.3': 'http://www.dmg.org/PMML-4_3'
}


def _validate_inputs(model, transformer, feature_names, target_values):
    logger.info('[x] Performing model validation.')
    if not type(model) in SUPPORTED_MODELS:
        raise TypeError("Provided model is not supported.")
    if not model.fit:
        raise TypeError("Provide a fitted model.")
    if transformer is not None and not type(transformer) in SUPPORTED_TRANSFORMERS:
        raise TypeError("Provided transformer is not supported.")
    if model.n_features_ != len(feature_names):
        logger.warning('[!] Network input shape does not match provided feature names - using generic names instead.')
        feature_names = ['x{}'.format(i) for i in range(model.n_features_)]
    if model.n_classes_ != len(target_values):
        logger.warning('[!] Network output shape does not match provided target values - using generic names instead.')
        target_values = ['y{}'.format(i) for i in range(model.n_classes_)]
    logger.info('[x] Model validation successful.')
    return feature_names, target_values


def _generate_header(root, kwargs):
    description = kwargs.get('description', None)
    copyright = kwargs.get('copyright', None)
    header = ET.SubElement(root, 'Header')
    if copyright:
        header.set('copyright', copyright)
    if description:
        header.set('description', description)
    timestamp = ET.SubElement(header, 'Timestamp')
    timestamp.text = str(datetime.now())
    return header


def _generate_data_dictionary(root, feature_names, target_name, target_values):
    data_dict = ET.SubElement(root, 'DataDictionary')
    data_field = ET.SubElement(data_dict, 'DataField')
    data_field.set('name', target_name)
    data_field.set('dataType', 'string')
    data_field.set('optype', 'categorical')
    logger.info('[x] Generating Data Dictionary:')
    for t in target_values:
        value = ET.SubElement(data_field, 'Value')
        value.set('value', t)
    for f in feature_names:
        data_field = ET.SubElement(data_dict, 'DataField')
        data_field.set('name', f)
        data_field.set('dataType', 'double')
        data_field.set('optype', 'continuous')
        logger.info('\t[-] {}...OK!'.format(f))
    return data_dict


def _generate_model(root, estimator, feature_names, target_name, target_names, model_name=None):
    model = _get_model(estimator)
    if type(model) == list:
        mining_model = ET.SubElement(root, 'MiningModel')
        mining_model.set('functionName', 'classification')
        if model_name:
            mining_model.set('modelName', model_name)
        _generate_mining_schema(mining_model, feature_names, target_name)
        _generate_output(mining_model, target_names)
        segmentation = ET.SubElement(mining_model, 'Segmentation')
        segmentation.set('multipleModelMethod', 'average')
        for i, m in enumerate(model):
            segment = ET.SubElement(segmentation, 'Segment')
            segment.set('id', str(i))
            ET.SubElement(segment, 'True')
            model_element = m.serialize(segment, feature_names, target_names)
            _generate_mining_schema(model_element, feature_names, target_name)
        return mining_model
    else:
        model_element = model.serialize(root, feature_names, target_names)
        _generate_mining_schema(model_element, feature_names, target_name)
        return model_element


def _generate_mining_schema(parent_element, feature_names, target_name):
    mining_schema = ET.SubElement(parent_element, 'MiningSchema')
    mining_field = ET.SubElement(mining_schema, 'MiningField')
    if target_name:
        mining_field.set('name', target_name)
        mining_field.set('usageType', 'target')
    for f in feature_names:
        mining_field = ET.SubElement(mining_schema, 'MiningField')
        mining_field.set('name', f)
        mining_field.set('usageType', 'active')
    return mining_schema


def _generate_output(parent_element, target_values):
    output = ET.SubElement(parent_element, 'Output')
    for t in target_values:
        output_field = ET.SubElement(output, 'OutputField')
        output_field.set('name', 'probability_{}'.format(t))
        output_field.set('feature', 'probability')
        output_field.set('value', t)
    return output


def _get_model(estimator):
    if type(estimator) == DecisionTreeClassifier:
        return TreeModel(estimator)
    if type(estimator) == RandomForestClassifier:
        return [TreeModel(tree) for tree in estimator.estimators_]


def sklearn2pmml(estimator, transformer=None, file=None, **kwargs):
    """
    Exports sklearn model as PMML.

    :param estimator: sklearn model to be exported as PMML (for supported models - see bellow).
    :param transformer: if provided then scaling is applied to data fields.
    :param file: name of the file where the PMML will be exported.
    :param kwargs: set of params that affects PMML metadata - see documentation for details.
    :return: XML element tree
    """

    feature_names = kwargs.get('feature_names', [])
    target_name = kwargs.get('target_name', 'class')
    target_values = kwargs.get('target_values', [])
    model_name = kwargs.get('model_name', None)
    pmml_version = kwargs.get('version', '4.2')

    feature_names, target_values = _validate_inputs(estimator, transformer, feature_names, target_values)

    pmml = ET.Element('PMML')
    pmml.set('version', pmml_version)
    pmml.set('xmlns', SUPPORTED_NS.get(pmml_version, 'http://www.dmg.org/PMML-4_2'))
    _generate_header(pmml, kwargs)
    _generate_data_dictionary(pmml, feature_names, target_name, target_values)
    _generate_model(pmml, estimator, feature_names, target_name, target_values, model_name)

    tree = ET.ElementTree(pmml)
    logger.info('[x] Generation of PMML successful.')
    if file:
        tree.write(file, encoding='utf-8', xml_declaration=True)
    return tree
