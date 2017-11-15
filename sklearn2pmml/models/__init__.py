try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Model:

    def __init__(self, estimator, pmml, function_name):
        self.estimator = estimator
        self.pmml = pmml
        self.function_name = function_name

    @property
    def n_features(self):
        raise NotImplementedError('Override for every model.')

    @property
    def n_classes(self):
        raise NotImplementedError('Override for every model.')

    @property
    def mining_schema(self):
        mining_schema = ET.Element('MiningSchema')
        mining_field = ET.SubElement(mining_schema, 'MiningField')
        if self.pmml.target_name:
            mining_field.set('name', self.pmml.target_name)
            mining_field.set('usageType', 'predicted')
        for f in self.pmml.feature_names:
            mining_field = ET.SubElement(mining_schema, 'MiningField')
            mining_field.set('name', f)
            mining_field.set('usageType', 'active')
        return mining_schema

    @property
    def local_transformations(self):
        transformer = self.pmml.transformer
        if transformer:
            local_transformations = ET.Element('LocalTransformations')
            for i, f in enumerate(self.pmml.feature_names):
                derived_field = ET.SubElement(local_transformations, 'DerivedField')
                derived_field.set('optype', 'continuous')
                derived_field.set('dataType', 'double')
                derived_field.set('name', '{}*'.format(f))
                if isinstance(transformer, StandardScaler):
                    if transformer.mean_[i] == 0:
                        norm_discrete = ET.SubElement(derived_field, 'NormDiscrete')
                        norm_discrete.set('field', f)
                        norm_discrete.set('value', '0.0')
                        logging.warning('[!] {field} has zero mean, avoiding scaling. Check whether your data does not contains only one value!'.format(field=f))
                    else:
                        norm_continuous = ET.SubElement(derived_field, 'NormContinuous')
                        norm_continuous.set('field', f)
                        ln1 = ET.SubElement(norm_continuous, 'LinearNorm')
                        ln2 = ET.SubElement(norm_continuous, 'LinearNorm')
                        ln1.set('orig', '0.0')
                        ln1.set('norm', (-transformer.mean_[i] / transformer.scale_[i]).astype(str))
                        ln2.set('orig', (transformer.mean_[i]).astype(str))
                        ln2.set('norm', '0.0')
                elif isinstance(transformer, MinMaxScaler):
                    norm_continuous = ET.SubElement(derived_field, 'NormContinuous')
                    norm_continuous.set('field', f)
                    ln1 = ET.SubElement(norm_continuous, 'LinearNorm')
                    ln2 = ET.SubElement(norm_continuous, 'LinearNorm')
                    ln1.set('orig', '0.0')
                    ln1.set('norm', (- transformer.min_[i] / (transformer.data_max_[i] - transformer.min_[i])).astype(str))
                    ln2.set('orig', (transformer.min_[i]).astype(str))
                    ln2.set('norm', '0.0')
            return local_transformations
        return None

    @property
    def output(self):
        output = ET.Element('Output')
        for t in self.pmml.target_values:
            output_field = ET.SubElement(output, 'OutputField')
            output_field.set('dataType', 'double')
            output_field.set('name', 'probability_{}'.format(t))
            output_field.set('feature', 'probability')
            output_field.set('value', t)
        return output
