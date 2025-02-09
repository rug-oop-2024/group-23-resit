from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import Lasso
from pydantic import PrivateAttr
from typing import Optional


class LassoWrapper(Model):
    """
    Linear regression wrapper that uses the lasso model
    and inherits its structure from the base model Model
    """
    _parameters: np.ndarray = PrivateAttr(default=None)

    def __init__(self) -> None:
        """Wraps the lasso model in this model"""
        self.model = Lasso()
        super().__init__(name="lasso_wrapper", type="regression")

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
        self.parameters = self.model.get_params()

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
