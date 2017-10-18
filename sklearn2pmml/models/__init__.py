try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


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
            mining_field.set('usageType', 'target' if self.function_name == 'classification' else 'predicted')
        for f in self.pmml.feature_names:
            mining_field = ET.SubElement(mining_schema, 'MiningField')
            mining_field.set('name', f)
            mining_field.set('usageType', 'active')
        return mining_schema

    @property
    def output(self):
        output = ET.Element('Output')
        for t in self.pmml.target_values:
            output_field = ET.SubElement(output, 'OutputField')
            output_field.set('name', 'probability_{}'.format(t))
            output_field.set('feature', 'probability')
            output_field.set('value', t)
        return output
