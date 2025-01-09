from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from pydantic import PrivateAttr
from typing import Literal
import os
import pickle
from copy import deepcopy


class Model(ABC):
    """Base model for all ml models"""
    _parameters = dict = PrivateAttr(default_factory=dict)
    name: str
    type = Literal["classification" or "regression"]

    def columns_of_ones(self, x: np.ndarray) -> np.ndarray:
        """Submethod for stacking column of 1's

        Arguments:
        x (np.ndarray)

        Returns:
        y (np.ndarray): resulting matrix after operation
        """
        ones_column = np.ones((x.shape[0], 1))
        y = np.hstack((x, ones_column))
        return y

    def to_artifact(self, name: str, asset_path: str =
                    "./model_artifacts/") -> Artifact:
        """Convert the model to an Artifact for storage or transfer."""
        os.makedirs(asset_path, exist_ok=True)

        # Serialize the model's attributes (e.g., parameters) to bytes
        # for storage
        model_data = {
            "name": self.name,
            "type": self.type,
            "parameters": self._parameters
        }
        model_bytes = pickle.dumps(model_data)

        artifact_asset_path = os.path.join(asset_path, f"{name}.pkl")

        with open(artifact_asset_path, 'wb') as f:
            f.write(model_bytes)

        # Construct and return the Artifact
        artifact = Artifact(
            name=name,
            asset_path=artifact_asset_path,
            data=model_bytes,
            type=self.type,
            tags=["model", self.type],
            metadata={"model_name": self.name, "model_type": self.type}
        )

        return artifact

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Training method blueprint for different linear regression models
        Return: None
        X represents the observations and y represents the ground_truth
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method blueprint for different linear regression models
        Return: ndarray representing a prediction
        """
        pass

    @property
    def parameters(self) -> dict:
        """
        Returns the copy of parameters in a dictionary
        """
        return deepcopy(self._parameters)
