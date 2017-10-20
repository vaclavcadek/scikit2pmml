try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from . import Model
from sklearn.tree import _tree
import numpy as np


class TreeModel(Model):

    def __init__(self, estimator, pmml, function_name):
        super(TreeModel, self).__init__(estimator, pmml, function_name)
        self.tree = self.estimator.tree_

    @property
    def n_features(self):
        return self.estimator.n_features_

    @property
    def n_classes(self):
        return self.estimator.n_classes_

    @property
    def model(self):

        def split(node_id, parent_id, operator, parent):
            node = ET.SubElement(parent, 'Node')
            node.set('id', str(node_id))
            left_child = self.tree.children_left[node_id]
            right_child = self.tree.children_right[node_id]
            values = self.tree.value[node_id][0]
            node.set('recordCount', str(sum(values)))
            node.set('score', self.pmml.target_values[np.argmax(values)])
            if operator:
                predicate = ET.SubElement(node, 'SimplePredicate')
                predicate.set('operator', operator)
                predicate.set('value', str(self.tree.threshold[parent_id]))
                predicate.set('field', self.pmml.feature_names[self.tree.feature[parent_id]])
            else:
                node.append(ET.Element('True'))
            for target_value, cnt_records in zip(self.pmml.target_values, values):
                score_distribution = ET.SubElement(node, 'ScoreDistribution')
                score_distribution.set('value', str(target_value))
                score_distribution.set('recordCount', str(cnt_records))
            if left_child != _tree.TREE_LEAF:
                split(left_child, node_id, 'lessOrEqual', node)
                split(right_child, node_id, 'greaterThan', node)

        tree_model = ET.Element('TreeModel')
        tree_model.append(self.mining_schema)
        tree_model.set('splitCharacteristic', 'binarySplit')
        tree_model.set('functionName', 'classification')
        split(0, 0, None, tree_model)
        return tree_model
