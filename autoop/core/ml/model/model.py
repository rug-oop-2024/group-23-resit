from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
import pickle
from copy import deepcopy


class Model(ABC):
    """Base model for all ml models"""

    def __init__(self, name: str, type: str) -> None:
        """Initialize the model"""
        self._parameters = {}
        self._name = name
        self._type = type

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

    def to_artifact(self, name: str,
                    base_path: str = "./models") -> Artifact:
        """Convert the model to an Artifact for storage or transfer."""

        # Serialize the model's attributes (e.g., parameters) to bytes
        # for storage
        model_data = {
            "name": self.name,
            "type": self.type,
            "parameters": self._parameters
        }
        model_bytes = pickle.dumps(model_data)

        artifact_asset_path = f"{name}.pkl"

        # Construct and return the Artifact
        artifact = Artifact(
            name=name,
            asset_path=artifact_asset_path,
            data=model_bytes,
            type=self.type,
            tags=["model", self.type],
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

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        """
        Setter for the parameters
        """
        self._parameters = parameters

    @property
    def type(self) -> str:
        """
        Returns the type of the model
        """
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        """
        Setter for the _type
        """
        self._type = type

    @property
    def name(self) -> str:
        """
        Returns the name of the model
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Setter for _name
        """
        self._name = name
