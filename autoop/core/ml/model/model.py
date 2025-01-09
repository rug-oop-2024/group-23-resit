from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from pydantic import PrivateAttr
from typing import Literal
import os
import pickle


class Model(ABC):
    """Base model for all ml models"""
    _parameters = dict = PrivateAttr(default_factory=dict)
    name: str
    type = Literal["classification" or "regression"]

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
