from autoop.core.ml.model import Model
import numpy as np


class RidgeRegression(Model):
    """
    Ridge Regression model for linear regression with L2 regularization.

    This model minimizes the least squares error with an added penalty
    proportional to the square of the magnitude of coefficients. The
    regularization term, controlled by alpha, helps prevent overfitting
    by discouraging large coefficients.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initializes the Ridge Regression model."""
        self.alpha = alpha  # Regularization strength
        self._parameters = {'coefficients': None, 'intercept': None}
        super().__init__(name="ridge_regression", type="regression")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the ridge regression model to the provided data using
        a closed-form solution.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Raises:
            ValueError: If the input data shapes are incompatible.
        """

        # Add a column of ones to observations for the intercept term
        x = self.columns_of_ones(x)

        # Closed-form solution of ridge regression:
        # (X.T * X + alpha * I)^(-1) * X.T * y
        n_features = x.shape[1]
        identity = np.eye(n_features)
        identity[0, 0] = 0

        # Ridge closed-form calculation
        inverse_term = np.linalg.inv(x.T @ x + self.alpha * identity)
        self._parameters['coefficients'] = inverse_term @ x.T @ y

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given feature matrix using
        the trained model.

        Args:
            x (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values as a vector.

        Raises:
            ValueError: If the model is not yet fitted or if input
            shapes are incompatible.
        """
        # Add a column of ones to observations for the intercept term
        x = self.columns_of_ones(x)

        # Predict using the linear combination of coefficients and observations
        return x @ self._parameters['coefficients']
