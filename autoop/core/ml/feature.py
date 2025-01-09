from typing import Literal
import numpy as np
from autoop.core.ml.dataset import Dataset
import pandas as pd


class Feature():
    """Feature class for the dataset's features"""

    def __init__(self, name: str,
                 type: Literal["numerical", "categorical"]) -> None:

        """
        Attributes: name: str = name of the feature in the dataset
                    type: Literal = numerical/categorical
        """
        self.name = name
        self.type = type

    def __str__(self) -> str:
        """Returns string of the name and feature type"""
        return f"Feature(name={self.name}, type={self.type})"

    def get_data(self, dataset: Dataset) -> np.ndarray:
        """Extracts the feature's data from the dataset as a NumPy array.

        Args:
            dataset (Dataset): The dataset from which to extract the feature
            data.

        Returns:
            np.ndarray: The feature data as a NumPy array.
        """
        df = dataset.read()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected dataset to return a DataFrame.")

        if self.name not in df.columns:
            raise ValueError(f"Feature '{self.name}' not found in dataset.")

        data = df[self.name].values
        if self.type == "categorical":
            data = pd.Categorical(data).codes
        return data
