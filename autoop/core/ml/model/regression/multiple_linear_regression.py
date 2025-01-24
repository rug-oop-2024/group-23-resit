from autoop.core.ml.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """
    Multiple linear regression model that inherits its methods structure and
    attributes from the base model Model
    """

    def __init__(self) -> None:
        """Inherits init from Model"""
        super().__init__(name="multiple_linear_regression", type="regression")

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
        if not np.all(X[:, 0] == 1):
            X = self.columns_of_ones(X)

        XtX = X.T @ X
        if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
            XtX_inv = np.linalg.pinv(XtX)
        else:
            XtX_inv = np.linalg.inv(XtX)

        self._parameters = {'parameters': XtX_inv @ X.T @ y}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Checks whether the new observations are in the right dimensions

        Creates a bias coloumn for the new oberservations

        Returns the vector of predictions by multiplying the new observations
        with the estimated paramaters
        """

        para = self._parameters['parameters']
        if not np.all(X[:, 0] == 1):
            new_matrix = self.columns_of_ones(X)
        return new_matrix @ para
