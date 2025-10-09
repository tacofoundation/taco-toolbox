from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample


class ValidationError(Exception):
    """
    Raised when sample validation fails.

    This exception is raised by SampleValidator implementations when
    a sample does not meet the required format specifications.

    Example:
        >>> raise ValidationError("TACOTIFF must use ZSTD compression")
    """

    pass


class SampleValidator(ABC):
    """
    Abstract base class for Sample validators.

    Validators enforce format-specific requirements on Sample objects.
    Each validator implements the validate() method to check if a sample
    meets specific format standards (e.g., TACOTIFF, TACOZARR, etc.).

    Validators are applied using the Sample.validate_with() method:
        >>> sample = Sample(id="x", path=Path("data.tif"), type="FILE")
        >>> sample.validate_with(TacotiffValidator())

    When creating a custom validator:
    1. Inherit from SampleValidator
    2. Implement validate() method
    3. Raise ValidationError with clear message if validation fails
    4. Optionally override get_supported_extensions() for auto-detection

    Example:
        >>> class MyFormatValidator(SampleValidator):
        ...     def validate(self, sample: Sample) -> None:
        ...         if sample.type != "FILE":
        ...             raise ValidationError("MyFormat requires FILE type")
        ...
        ...         # Check format-specific requirements
        ...         if not self._is_valid_format(sample.path):
        ...             raise ValidationError("Invalid MyFormat file")
        ...
        ...     def get_supported_extensions(self) -> list[str]:
        ...         return [".myformat", ".mf"]
    """

    @abstractmethod
    def validate(self, sample: "Sample") -> None:
        """
        Validate a sample against format-specific requirements.

        This method should perform all necessary checks to ensure the sample
        meets the format specification. If validation fails, it must raise
        a ValidationError with a clear, descriptive message.

        Args:
            sample: Sample object to validate

        Raises:
            ValidationError: If sample does not meet format requirements

        Example:
            >>> def validate(self, sample: Sample) -> None:
            ...     if sample.type != "FILE":
            ...         raise ValidationError("Expected FILE type")
            ...
            ...     # Perform format checks...
            ...     if not self._check_format(sample.path):
            ...         raise ValidationError("Invalid format structure")
        """
        pass

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of file extensions supported by this validator.

        This method can be used for auto-detection of appropriate validators
        based on file extensions. By default, returns an empty list (no
        auto-detection).

        Returns:
            List of file extensions (including the dot, e.g., [".tif", ".tiff"])

        Example:
            >>> def get_supported_extensions(self) -> list[str]:
            ...     return [".tif", ".tiff", ".gtiff"]
        """
        return []

    def __repr__(self) -> str:
        """String representation of the validator."""
        return f"{self.__class__.__name__}()"
