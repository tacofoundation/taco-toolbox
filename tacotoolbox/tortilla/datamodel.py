from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl
import pydantic


if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample

class TortillaExtension(ABC, pydantic.BaseModel):
    """Abstract base class for Tortilla extensions that add computed columns."""

    return_none: bool = pydantic.Field(False, description="If True, return None values while preserving schema")

    @abstractmethod
    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        pass

    @abstractmethod
    def _compute(self, tortilla: "Tortilla") -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
        pass

    def __call__(self, tortilla: "Tortilla") -> pl.DataFrame:
        """
        Process Tortilla and return computed metadata.

        Args:
            tortilla: Input Tortilla object

        Returns:
            pl.DataFrame: DataFrame with computed metadata
        """
        # Check return_none FIRST for performance
        if self.return_none:
            schema = self.get_schema()
            none_data = {col_name: [None] for col_name in schema}
            return pl.DataFrame(none_data, schema=schema)

        # Only do actual computation if needed
        return self._compute(tortilla)


class Tortilla:
    """
    Container for Sample collections with hierarchical metadata export.

    Supports nested sample structures up to 5 levels deep with position tracking.
    Uses eager DataFrame construction with schema validation for performance.
    """

    def __init__(self, samples: list["Sample"]) -> None:
        """
        Create Tortilla with eager DataFrame construction and schema validation.

        Args:
            samples: List of Sample objects

        Raises:
            ValueError: If samples have inconsistent metadata schemas
        """
        if not samples:
            raise ValueError("Cannot create Tortilla with empty samples list")

        self.samples = samples

        # Extract metadata from all samples
        metadata_dfs = []
        reference_columns = None

        for i, sample in enumerate(samples):
            sample_metadata_df = sample.export_metadata()

            # Validate schema consistency
            if reference_columns is None:
                reference_columns = set(sample_metadata_df.columns)
            else:
                current_columns = set(sample_metadata_df.columns)

                if current_columns != reference_columns:
                    missing_columns = reference_columns - current_columns
                    extra_columns = current_columns - reference_columns

                    error_msg = f"Schema inconsistency at sample {i} (id: {sample.id}):\n"

                    if missing_columns:
                        error_msg += f"  Missing columns: {sorted(missing_columns)}\n"

                    if extra_columns:
                        error_msg += f"  Extra columns: {sorted(extra_columns)}\n"

                    error_msg += f"  Expected schema: {sorted(reference_columns)}\n"
                    error_msg += f"  Actual schema: {sorted(current_columns)}"

                    raise ValueError(error_msg)

            metadata_dfs.append(sample_metadata_df)

        # Concatenate DataFrames - all schemas are guaranteed to be consistent
        self._metadata_df = pl.concat(metadata_dfs, how="vertical")
        self._current_depth = self._calculate_current_depth()

    def _calculate_current_depth(self) -> int:
        """Calculate current depth by examining first sample."""
        if not self.samples:
            return 0

        first_sample = self.samples[0]

        if hasattr(first_sample, "path") and hasattr(first_sample.path, "samples") and first_sample.path.samples:
            child_tortilla = Tortilla(first_sample.path.samples)
            return 1 + child_tortilla._current_depth
        else:
            return 0

    def __len__(self) -> int:
        """Return number of samples in this tortilla."""
        return len(self.samples)

    def extend_with(self, extension: TortillaExtension) -> "Tortilla":
        """
        Add extension data via DataFrame processing.

        Args:
            extension: Extension implementing TortillaExtension ABC

        Returns:
            Self for method chaining

        Raises:
            ValueError: If row count mismatch or column conflicts
        """
        result_df = extension(self)

        if len(result_df) != len(self._metadata_df):
            raise ValueError(
                f"Extension returned {len(result_df)} rows, expected {len(self._metadata_df)} (one per sample)."
            )

        conflicts = set(result_df.columns) & set(self._metadata_df.columns)
        if conflicts:
            raise ValueError(f"Column conflicts: {sorted(conflicts)}")

        # Horizontal concatenation in polars using hstack
        self._metadata_df = self._metadata_df.hstack(result_df)
        return self

    def export_metadata(self, deep: int = 0) -> pl.DataFrame:
        """
        Export metadata with optional hierarchical expansion.

        Args:
            deep: Expansion level (0-5 max)
                - 0: Current level only (with extensions)
                - >0: Expand N levels deep (base metadata only, adds internal:parent_id)

        Returns:
            DataFrame with sample metadata and optional position tracking
        """
        if deep < 0:
            raise ValueError("Deep level must be non-negative")

        if deep > 5:
            raise ValueError("Maximum depth is 5")

        if deep == 0:
            return self._metadata_df.clone()

        return self._expand_hierarchical(deep)

    def _expand_hierarchical(self, target_deep: int) -> pl.DataFrame:
        """
        Build expanded DataFrame with linked-list position encoding.

        Args:
            target_deep: Target expansion depth

        Returns:
            DataFrame with expanded samples and internal:parent_id column (linked-list style)
        """
        if target_deep > self._current_depth:
            raise ValueError(f"Cannot expand to depth {target_deep}, current max depth is {self._current_depth}")

        current_samples = self.samples
        current_dfs = []

        for _level in range(1, target_deep + 1):
            next_dfs = []
            next_samples = []

            for parent_idx, sample in enumerate(current_samples):
                if hasattr(sample, "path") and hasattr(sample.path, "samples") and sample.path.samples:
                    for child_sample in sample.path.samples:
                        child_metadata_df = child_sample.export_metadata()

                        # Add position column pointing to parent in previous level
                        child_metadata_df = child_metadata_df.with_columns(
                            pl.lit(parent_idx).alias("internal:parent_id")
                        )

                        next_dfs.append(child_metadata_df)
                        next_samples.append(child_sample)

            current_dfs = next_dfs
            current_samples = next_samples

        return pl.concat(current_dfs, how="vertical")
