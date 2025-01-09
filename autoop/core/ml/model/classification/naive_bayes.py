from autoop.core.ml.model import Model
import numpy as np
from copy import deepcopy
from pydantic import PrivateAttr


class NaiveBayesModel(Model):
    """
    Naive Bayes classifier for binary or multi-class classification.

    This model assumes features follow a Gaussian (Normal) distribution and
    uses Bayes' theorem to calculate the posterior probability for each class.
    It is suitable for applications in binary or multi-class classification
    where independence of features is assumed.
    """
    _parameters = dict = PrivateAttr(default_factory=dict)
    type: str = "classification"
    name: str = "naive_bayes"

    def __init__(self) -> None:
        """Initialization"""
        self._parameters = {
            'priors': None,
            'likelihoods': None

        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Naive Bayes classifier by calculating priors and likelihoods
        based on the provided training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Raises:
            ValueError: If input data shapes are incompatible.

        The model computes:
            - Priors: The probability of each class occurring in the data.
            - Likelihoods: The mean and variance of each feature for
            each class,
              assuming a Gaussian distribution.
        """

        self.classes = np.unique(y)
        self._parameters['priors'] = {}
        self._parameters['likelihoods'] = {}

        for c in self.classes:
            X_c = X[y == c]
            self._parameters['priors'][c] = X_c.shape[0] / X.shape[0]
            # Calculating the mean and variance for Gaussian Naive Bayes
            self._parameters['likelihoods'][c] = {
                'mean': X_c.mean(axis=0),
                # Adding a small constant for numerical stability:
                'var': X_c.var(axis=0) + 1e-9
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given feature matrix using the
        trained model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.


        Prediction Process:
            - For each input sample, calculates the posterior log-probability
              of each class
              using the priors and Gaussian likelihoods for each feature.
            - Chooses the class with the highest posterior log-probability.
        """
        predictions = []
        for x in X:
            class_probabilities = {}
            for c in self.classes:
                # Prior log-probability
                prior = np.log(self._parameters['priors'][c])
                # Likelihood log-probability under Gaussian assumption
                variance = self._parameters['likelihoods'][c]['var']
                mean_diff = x - self._parameters['likelihoods'][c]['mean']

                likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * variance) + (mean_diff ** 2) / variance
                )
                # Total log-probability for class c
                class_probabilities[c] = prior + likelihood
            # Choose class with the highest log-probability
            pred_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(pred_class)
        return np.array(predictions)

    @property
    def get_parameters(self) -> dict:
        """
        Return the copy of learned parameters (observations and ground truth)
        as a dictionary.

        """
        return deepcopy(self._parameters)
