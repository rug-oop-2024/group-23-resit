from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import Lasso
from pydantic import PrivateAttr
from copy import deepcopy
from typing import Optional


class LassoWrapper(Model):
    """
    Linear regression wrapper that uses the lasso model
    and inherits its structure from the base model Model
    """
    _parameters: np.ndarray = PrivateAttr(default=None)
    name: str = "lasso_wrapper"
    type: str = "regression"

    def __init__(self) -> None:
        """Wraps the lasso model in this model"""
        self.model = Lasso()

    def fit(self, x: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            check_input: bool = True) -> None:
        """
        Fit the Lasso model to the input data X and target y.
        Parameters:
        X (np.ndarray): Input feature matrix (n_samples, n_features)
        y (np.ndarray): Target values (n_samples,)
        """

        self.model.fit(x, y, sample_weight=sample_weight,
                       check_input=check_input)
        self.parameters = np.append(self.model.intercept_, self.model.coef_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input data X using
        the fitted model.

        Parameters:
        X (np.ndarray): Input feature matrix (n_samples, n_features)

        Returns:
        prediction (np.ndarray): Predicted target values (n_samples)
        """

        if self._parameters is None:
            raise ValueError("Model has not been trained yet."
                             "Call fit first.")
        prediction = self.model.predict(x)
        return prediction

    @property
    def get_parameters(self) -> dict:
        """
        Return the copy of learned parameters (intercept and coefficients)
        as a dictionary.
        Returns:
        dict: copy of Dictionary with keys 'intercept' and 'coefficients'
        """
        return deepcopy({
            'intercept': self._parameters[0],
            'coefficients': self._parameters[1:]
        })
