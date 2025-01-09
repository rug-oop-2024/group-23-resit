"""Models"""
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.regression.lasso_wrapper import LassoWrapper
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors)
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegressionModel)
from autoop.core.ml.model.classification.naive_bayes import NaiveBayesModel

REGRESSION_MODELS = ["multiple_linear_regression",
                     "lasso_wrapper",
                     "ridge_regression"
                     ]

CLASSIFICATION_MODELS = ["knn",
                         "logistic_regression",
                         "naive_bayes"
                         ]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""

    models = {
        "multiple_linear_regression": MultipleLinearRegression(),
        "lasso_wrapper": LassoWrapper(),
        "ridge_regression": RidgeRegression(),
        "knn": KNearestNeighbors(),
        "logistic_regression": LogisticRegressionModel(),
        "naive_bayes": NaiveBayesModel()
    }
    return models.get(model_name, None)
