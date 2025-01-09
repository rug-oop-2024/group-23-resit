import base64
from typing import Dict, List, Optional


class Artifact:
    """Artifact"""
    def __init__(self, name: str, asset_path: str, data: bytes,
                 type: str, tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None,
                 version: str = "1.0.0") -> None:
        """
        Attributes:
                  name: str
                  asset_path: str
                  data: bytes
                  version: str
                  type: str
                  tags: Optional[List[str]]
                  metadata: Optional[Dict]
        """
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.type = type
        self.tags = tags
        self.metadata = metadata

    @property
    def id(self) -> str:
        """Derives the ID from the asset_path and version."""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        safe_version = self.version.replace(".",
                                            "_").replace(":",
                                                         "_").replace("=", "_")
        encoded_path = encoded_path[:-2]
        return f"{encoded_path}_{safe_version}"

    def read(self) -> bytes:
        """Returns the artifact's data."""
        return self.data

    def save(self, new_data: bytes) -> None:
        """Set or update the artifact's data."""
        self._data = new_data

    def get(self, attribute: str) -> str | bytes:
        """Retrieve an attribute by name if it exists, otherwise
           raise AttributeError.
        Args:
            attribute (str): The name of the attribute to retrieve.
        Returns:
            Any: The value of the requested attribute.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has "
                                 f"no attribute '{attribute}'")

    def get_metadata(self) -> Dict:
        """Returns metadata of the artifact."""
        return {
            "name": self.name,
            "version": self.version,
            "asset_path": self.asset_path,
            "tags": self.tags,
            "metadata": self.metadata,
            "type": self.type
        }
