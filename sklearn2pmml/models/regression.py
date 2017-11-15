try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from . import Model
import numpy as np


class RegressionModel(Model):

    LINEAR_REGRESSION = 'linearRegression'
    LOGISTIC_REGRESSION = 'logisticRegression'

    def __init__(self, estimator, pmml, function_name, type):
        super(RegressionModel, self).__init__(estimator, pmml, function_name)
        self.type = type

    @property
    def n_features(self):
        return self.estimator.coef_.size

    @property
    def n_classes(self):
        return self.estimator.classes_.size

    @property
    def model(self):
        regression_model = ET.Element('RegressionModel')
        regression_model.set('functionName', self.function_name)
        if self.type == RegressionModel.LOGISTIC_REGRESSION:
            regression_model.set('normalizationMethod', 'logit')
        if self.pmml.model_name:
            regression_model.set('modelName', self.pmml.model_name)
        regression_model.append(self.mining_schema)
        if self.function_name == 'classification':
            regression_model.append(self.output)
        if self.local_transformations:
            regression_model.append(self.local_transformations)
        for i, target_value in enumerate(reversed(self.pmml.target_values)):
            regression_table = ET.SubElement(regression_model, 'RegressionTable')
            regression_table.set('targetCategory', target_value)
            if i == len(self.pmml.target_values) - 1:
                intercept = '0'
            else:
                intercept = np.atleast_1d(self.estimator.intercept_).astype(str)[0]
                for feature, coefficient in zip(self.pmml.feature_names, self.estimator.coef_.astype(str).ravel()):
                    numeric_predictor = ET.SubElement(regression_table, 'NumericPredictor')
                    numeric_predictor.set('name', '{}*'.format(feature) if self.local_transformations else feature)
                    numeric_predictor.set('coefficient', coefficient)
            regression_table.set('intercept', intercept)
        return regression_model
