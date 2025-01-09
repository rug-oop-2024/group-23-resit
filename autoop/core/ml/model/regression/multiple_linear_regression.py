from autoop.core.ml.model import Model
import numpy as np
from copy import deepcopy


class MultipleLinearRegression(Model):
    """
    Multiple linear regression model that inherits its methods structure and
    attributes from the base model Model
    """
    type: str = "regression"
    name: str = "multiple_linear_regression"

    def __init__(self) -> None:
        """Inherits init from Model"""
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data using Equation 11 from the instructions.

        Arguments:
        observations (np.ndarray): The observation matrix, with samples
        as rows and variables as columns.
        ground_truth (np.ndarray): The ground truth vector.

        Stores the fitted parameters in the 'parameters'
        attribute as a dictionary.
        """

        # Adds a column of ones to observations
        observations_tilde = np.column_stack([np.ones(X.shape[0]),
                                             X])

        est_param = (np.linalg.pinv(observations_tilde.T @ observations_tilde)
                     @ observations_tilde.T @ y)

        self._parameters = {'parameters': est_param}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Checks whether the new observations are in the right dimensions

        Creates a bias coloumn for the new oberservations

        Returns the vector of predictions by multiplying the new observations
        with the estimated paramaters
        """

        para = self._parameters['parameters']

        new_matrix = np.c_[np.ones(X.shape[0]),
                           X]
        return new_matrix @ para

    @property
    def get_parameters(self) -> dict:
        """
        Returns the copy of parameters in a dictionary
        """
        return deepcopy(self._parameters)
