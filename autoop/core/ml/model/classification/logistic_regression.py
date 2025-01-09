from autoop.core.ml.model import Model
import numpy as np
from copy import deepcopy


class LogisticRegressionModel(Model):
    """
    Logistic regression model for binary or multi-class classification.

    This model uses logistic regression to estimate probabilities and
    make predictions for classification tasks. It is optimized using
    gradient descent to minimize the logistic loss function.

    """
    type: str = "classification"
    name: str = "logistic_regression"

    def __init__(self, learning_rate: float = 0.01,
                 iterations: int = 1000) -> None:
        """
        Initializes the Logistic Regression model with specified learning rate
        and number of iterations.

        Args:
            learning_rate (float, optional): The step size for gradient
            descent. Default is 0.01.
            iterations (int, optional): The number of iterations for gradient
            descent. Default is 1000.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._parameters = {
            'weights': None
        }

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function to the input.

        Args:
            z (np.ndarray): Input values (linear combination of weights
            and features).

        Returns:
            np.ndarray: Transformed values in the range (0, 1), representing
            probabilities.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the logistic regression model to the provided training data using
        gradient descent.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        The model adds a bias term to each observation, initializes weights
        to zero, and iteratively updates weights based on the gradient of the
        loss function.
        """
        # Adding a bias term to the observations
        X = np.c_[np.ones(X.shape[0]), X]
        if y.ndim > 1:
            y = y.flatten()

        # Initializing weights
        self._parameters['weights'] = np.zeros(X.shape[1])

        # Gradient descent
        for _ in range(self.iterations):
            # Compute predictions
            predictions = self.sigmoid(X @ self._parameters['weights'])
            # Update weights
            gradient = X.T @ (predictions - y) / y.size
            self._parameters['weights'] -= self.learning_rate * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given feature matrix.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels (0 or 1) for binary
            classification.

        The method applies the sigmoid function to the linear combination
        of weights and features, then converts probabilities to binary
        predictions (0 or 1) based on a threshold of 0.5.
        """
        # Adding a bias term to the observations
        X = np.c_[np.ones(X.shape[0]), X]
        # Predicting probabilities
        probabilities = self.sigmoid(X @ self._parameters['weights'])
        # Converting probabilities to binary class predictions
        return (probabilities >= 0.5).astype(int)

    @property
    def get_parameters(self) -> dict:
        """
        Return the copy of learned parameters (observations and ground truth)
        as a dictionary.

        """
        return deepcopy(self._parameters)
