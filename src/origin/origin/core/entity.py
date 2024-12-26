from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Entity(ABC):
    """
    Abstract base class for biological entities in the Origin library.

    This class serves as the foundation for all entities (e.g., Molecule, Protein, RNA).
    Each entity can store metadata, provide a unified interface for data representation,
    and ensure compatibility across different downstream tasks.

    Attributes:
        id (str): Unique identifier for the entity.
        name (Optional[str]): Human-readable name of the entity.
        metadata (Dict[str, Any]): Additional metadata or annotations for the entity.
    """

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an Entity.

        Args:
            id (str): Unique identifier for the entity.
            name (Optional[str]): Human-readable name of the entity.
            metadata (Optional[Dict[str, Any]]): Additional metadata or annotations for the entity.
        """
        self.id = id
        self.name = name
        self.metadata = metadata or {}

    @abstractmethod
    def summary(self) -> str:
        """
        Provide a textual summary of the entity.
        This method must be implemented by subclasses.

        Returns:
            str: A summary string describing the entity.
        """
        pass

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Add or update a metadata field for the entity.

        Args:
            key (str): Metadata key.
            value (Any): Metadata value.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a metadata value.

        Args:
            key (str): Metadata key.
            default (Optional[Any]): Default value to return if the key is not found.

        Returns:
            Any: Metadata value or default if key is not found.
        """
        return self.metadata.get(key, default)

    def __str__(self) -> str:
        """
        String representation of the entity.

        Returns:
            str: A string describing the entity.
        """
        return f"Entity(id={self.id}, name={self.name}, metadata={self.metadata})"

    def __repr__(self) -> str:
        """
        Debug representation of the entity.

        Returns:
            str: A detailed string representation of the entity.
        """
        return f"{self.__class__.__name__}(id={self.id}, name={self.name}, metadata={self.metadata})"
