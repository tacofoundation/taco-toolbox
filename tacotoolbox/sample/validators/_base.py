from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample


class ValidationError(Exception):
    """Raised when sample validation fails."""

    pass


class SampleValidator(ABC):
    """
    Abstract base class for Sample validators.

    Validators enforce format-specific requirements on Sample objects.
    Applied using Sample.validate_with() method.

    When creating a custom validator:
    1. Inherit from SampleValidator
    2. Implement validate() method
    3. Raise ValidationError with clear message if validation fails
    4. Optionally override get_supported_extensions() for auto-detection
    """

    @abstractmethod
    def validate(self, sample: "Sample") -> None:
        """
        Validate sample against format-specific requirements.

        Must raise ValidationError if sample doesn't meet format requirements.
        """
        pass

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of file extensions supported by this validator.

        Used for auto-detection of appropriate validators based on file extensions.
        Returns empty list by default (no auto-detection).
        """
        return []

    def __repr__(self) -> str:
        """String representation of the validator."""
        return f"{self.__class__.__name__}()"
