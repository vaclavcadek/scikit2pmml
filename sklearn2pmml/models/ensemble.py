try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from . import Model
from sklearn2pmml import TreeModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


class Segmentation(Model):

    def __init__(self, estimator, pmml, function_name):
        super(Segmentation, self).__init__(estimator, pmml, function_name)
        if type(estimator) in [RandomForestClassifier, ExtraTreesClassifier]:
            self.serializers = [TreeModel(tree, pmml, function_name) for tree in self.estimator.estimators_]

    @property
    def n_features(self):
        return self.estimator.n_features_

    @property
    def n_classes(self):
        return self.estimator.n_classes_

    @property
    def model(self):
        mining_model = ET.Element('MiningModel')
        mining_model.set('functionName', self.function_name)
        if self.pmml.model_name:
            mining_model.set('modelName', self.pmml.model_name)
        mining_model.append(self.mining_schema)
        mining_model.append(self.output)
        segmentation = ET.SubElement(mining_model, 'Segmentation')
        segmentation.set('multipleModelMethod', 'average')
        for i, serializer in enumerate(self.serializers):
            segment = ET.SubElement(segmentation, 'Segment')
            segment.set('id', str(i))
            segment.append(ET.Element('True'))
            segment.append(serializer.model)
        return mining_model
