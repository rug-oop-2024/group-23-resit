from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Dataset inheriting from Artifact"""

    def __init__(self, *args, **kwargs) -> None:
        """Inherits initialization from Artifact"""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0") -> 'Dataset':
        """Returns a Dataset from a dataframe"""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Read the dataset from the saved CSV data."""

        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save a DataFrame as CSV data in the artifact."""

        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
