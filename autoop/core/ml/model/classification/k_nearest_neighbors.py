from autoop.core.ml.model import Model
import numpy as np
from pydantic import Field, PrivateAttr, field_validator
from collections import Counter


class KNearestNeighbors(Model):
    """
    Classification Model that makes predictions based on the
    k closest neighbors
    Inherits its structure from the base model Model
    """
    k: int = Field(title="Number of neighbors", default=3)
    _parameters = dict = PrivateAttr(default_factory=dict)

    def __init__(self, k: int = 3) -> None:
        """Initialize with default k=3"""
        self.k = k
        super().__init__(name="k_nearest_neighbors", type="classification")

    @field_validator("k")
    def k_greater_than_zero(cls, value: int) -> int:
        """Field validator for k. It should be an integer greater than 0"""
        if value <= 0:
            raise ValueError("k must be greater than 0")
        return value

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model by fitting the observations and ground_truth into the
        _parameters dictionary
        """
        # Fixing the shape of y
        if y.ndim > 1:
            if y.shape[1] > 1:
                y = np.argmax(y, axis=1)
        else:
            y = y.flatten()

        self._parameters = {
            "observations": x,
            "ground_truth": y
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the new observations using a helper method
        """
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with"
                             "appropriate arguments")

        predictions = [self._predict_single(x1) for x1 in x]
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray) -> list:
        """
        Helps the predict method by making a prediction for a single
        observation
        """

        distances = np.linalg.norm(self._parameters["observations"] - x,
                                   axis=1)
        if not isinstance(self.k, int):
            raise TypeError(f"k must be an integer, got {type(self.k)}")

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i]
                            for i in k_indices]
        most_common = Counter(k_nearest_labels). most_common()
        return most_common[0][0]
