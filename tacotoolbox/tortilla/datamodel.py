"""
Tortilla datamodel for TACO framework.

Tortilla is a container for Sample collections with hierarchical metadata export.
Supports nested sample structures with position tracking.

Key features:
- Hierarchical metadata export (up to SHARED_MAX_DEPTH levels)
- Position-Isomorphic Tree (PIT) validation
- Auto-padding for homogeneous structures
- Efficient DataFrame construction with schema validation
- Configurable schema strictness (strict_schema parameter)

Best Practice: When mixing FOLDERs and FILEs, place all FOLDERs before FILEs
for better organization and readability.
"""

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, cast

import polars as pl
import pydantic

from tacotoolbox._column_utils import align_dataframe_schemas
from tacotoolbox._constants import (
    METADATA_PARENT_ID,
    PADDING_PREFIX,
)
from tacotoolbox._utils import validate_depth


if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample


class TortillaExtension(ABC, pydantic.BaseModel):
    """Abstract base class for Tortilla extensions that add computed columns."""

    schema_only: bool = pydantic.Field(
        False,
        description="If True, return None values while preserving schema",
        validation_alias="return_none",  # Backward compatibility
    )

    @abstractmethod
    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        pass

    @abstractmethod
    def _compute(self, tortilla: "Tortilla") -> pl.DataFrame:
        """Actual computation logic - only called when schema_only=False."""
        pass

    def __call__(self, tortilla: "Tortilla") -> pl.DataFrame:
        """Process Tortilla and return computed metadata."""
        # Check schema_only FIRST for performance
        if self.schema_only:
            schema = self.get_schema()
            none_data = {col_name: [None] for col_name in schema}
            return pl.DataFrame(none_data, schema=schema)

        # Only do actual computation if needed
        return self._compute(tortilla)


class Tortilla:
    """
    Container for Sample collections with hierarchical metadata export.

    Supports nested sample structures with position tracking (max depth from constants).
    Uses eager DataFrame construction with schema validation for performance.

    Best Practice: When mixing FOLDERs and FILEs in the same Tortilla,
    place all FOLDERs before FILEs for better organization and readability.

    Auto-Padding: Use `pad_to` parameter to automatically pad sample list to
    make length divisible by a specific value. Useful for creating homogeneous
    tree structures. Padding samples have IDs starting with "__TACOPAD__" and
    preserve the schema of real samples with None values.

    Schema Validation: By default (strict_schema=True), all samples must have
    identical metadata schemas. Set strict_schema=False to allow heterogeneous
    schemas with automatic None-filling for missing columns.
    """

    def __init__(
        self,
        samples: list["Sample"],
        pad_to: int | None = None,
        strict_schema: bool = True,
    ) -> None:
        """
        Create Tortilla with eager DataFrame construction and schema validation.

        Raises:
            ValueError: If samples have inconsistent metadata schemas (strict_schema=True)
                       or empty list or if Inclusive ID Rule is violated or duplicate IDs
        """
        if not samples:
            raise ValueError("Cannot create Tortilla with empty samples list")

        # Auto-pad if requested
        if pad_to is not None:
            samples = self._create_padded_samples(samples, pad_to)

        self.samples = samples

        # Validate unique IDs at this level
        self._validate_unique_ids()

        # Check ordering best practice (only for mixed types)
        self._check_sample_ordering()

        # Validate Inclusive ID Rule
        self._validate_inclusive_ids()

        # Extract metadata from all samples
        metadata_dfs = []
        for sample in samples:
            metadata_dfs.append(sample.export_metadata())

        # Handle schema validation/alignment
        if strict_schema:
            # STRICT MODE: All schemas must match exactly
            reference_columns = set(metadata_dfs[0].columns)
            for i, df in enumerate(metadata_dfs[1:], start=1):
                current_columns = set(df.columns)
                if current_columns != reference_columns:
                    self._raise_schema_mismatch_error(
                        i, samples[i], reference_columns, current_columns
                    )
        else:
            # FLEXIBLE MODE: Use helper function to align schemas
            metadata_dfs = align_dataframe_schemas(
                metadata_dfs, core_fields=["id", "type", "path"]
            )

        # Concatenate DataFrames - all schemas are guaranteed consistent
        self._metadata_df = pl.concat(metadata_dfs, how="vertical")
        self._current_depth = self._calculate_current_depth()

    def _raise_schema_mismatch_error(
        self,
        sample_index: int,
        sample: "Sample",
        reference_columns: set[str],
        current_columns: set[str],
    ) -> None:
        """Raise helpful error when schema mismatch detected in strict mode."""
        missing_columns = reference_columns - current_columns
        extra_columns = current_columns - reference_columns

        error_msg = f"Schema inconsistency detected at sample {sample_index} (id: '{sample.id}'):\n\n"

        error_msg += (
            f"  Reference sample columns: {sorted(reference_columns)}\n"
            f"  Current sample columns: {sorted(current_columns)}\n\n"
        )

        if missing_columns:
            error_msg += f"  Missing columns: {sorted(missing_columns)}\n"

        if extra_columns:
            error_msg += f"  Extra columns: {sorted(extra_columns)}\n"

        error_msg += (
            "\n"
            "  💡 Hint: If you want to combine samples with different schemas, use:\n"
            "      Tortilla(samples=[...], strict_schema=False)\n\n"
            "  This will auto-fill missing columns with None values.\n\n"
            "  To inspect sample schemas before creating Tortilla:\n"
            "      df = sample.export_metadata()\n"
            "      print(df.columns)  # See what columns exist\n"
        )

        raise ValueError(error_msg)

    def _align_schema(self, df: pl.DataFrame, target_columns: set[str]) -> pl.DataFrame:
        """
        Align DataFrame schema to match target columns by adding missing columns with None.
        """
        current_columns = set(df.columns)
        missing_columns = target_columns - current_columns

        if not missing_columns:
            return df

        # Add missing columns with None values
        for col_name in missing_columns:
            df = df.with_columns(pl.lit(None).alias(col_name))

        return df

    def _validate_unique_ids(self) -> None:
        """
        Ensure all sample IDs are unique at this level.

        Duplicate IDs cause silent failures in ZIP offset calculation because
        offsets are stored in a dictionary keyed by archive path. When duplicate
        IDs exist, later samples overwrite earlier ones in the offset map,
        resulting in missing files in the final container.
        """
        ids = [s.id for s in self.samples]
        duplicates = {id: count for id, count in Counter(ids).items() if count > 1}

        if duplicates:
            # Show first 10 duplicates for debugging
            dup_list = ", ".join(
                f"'{id}' ({count}x)" for id, count in list(duplicates.items())[:10]
            )

            raise ValueError(
                f"Duplicate sample IDs found in Tortilla: {dup_list}\n"
                f"Total unique IDs with duplicates: {len(duplicates)}\n"
                f"Total samples: {len(ids)}\n"
                f"Each sample at the same level must have a unique ID.\n"
            )

    @staticmethod
    def _create_padded_samples(samples: list, pad_to: int) -> list:
        """
        Create padding samples to make total length divisible by pad_to.

        Padding samples use empty bytes (b"") which creates 0-byte temporary files.
        These files can be copied to ZIP/FOLDER containers and are automatically
        cleaned up after container creation.

        Padding IDs are sequential: __TACOPAD__0, __TACOPAD__1, __TACOPAD__2, etc.
        """
        # Local import to avoid circular dependency
        from tacotoolbox.sample.datamodel import Sample

        # Check if padding needed
        if len(samples) % pad_to == 0:
            return samples

        # Calculate number of padding samples needed
        num_padding = pad_to - (len(samples) % pad_to)

        # Get reference schema from first sample
        ref_df = samples[0].export_metadata()

        # Create padding samples
        padded_samples = samples.copy()

        for i in range(num_padding):
            # Create padding sample using internal factory
            # This creates a 0-byte temp file that bypasses ID validation
            dummy = Sample._create_padding(index=i)

            # Get extension columns (excluding core fields)
            extension_cols = [
                col for col in ref_df.columns if col not in ["id", "type", "path"]
            ]

            if extension_cols:
                # Create DataFrame with None values using reference schema
                schema = {col: ref_df.schema[col] for col in extension_cols}
                none_data = {col: [None] for col in extension_cols}
                padding_df = pl.DataFrame(none_data, schema=schema)

                # Extend dummy with padding DataFrame
                dummy.extend_with(padding_df)

            padded_samples.append(dummy)

        return padded_samples

    def _check_sample_ordering(self) -> None:
        """Check if samples follow best practice: FOLDERs before FILEs."""
        types = [sample.type for sample in self.samples]
        has_folders = "FOLDER" in types
        has_files = "FILE" in types

        if not (has_folders and has_files):
            return

        first_folder_idx = next(
            (i for i, sample in enumerate(self.samples) if sample.type == "FOLDER"),
            None,
        )
        first_file_idx = next(
            (i for i, sample in enumerate(self.samples) if sample.type == "FILE"), None
        )

        if first_file_idx is not None and first_folder_idx is not None:
            if first_file_idx < first_folder_idx:
                warnings.warn(
                    f"Consider placing FOLDERs before FILEs for better organization. "
                    f"Found FILE at position {first_file_idx} before FOLDER at position {first_folder_idx}.",
                    UserWarning,
                    stacklevel=3,
                )

    def _validate_inclusive_ids(self) -> None:
        """
        Validate Inclusive ID Rule: The folder with maximum non-padding IDs
        must contain ALL IDs from folders with fewer IDs (subset relationship).

        IDs starting with '__TACOPAD__' are IGNORED in validation as they
        represent padding placeholders, not real data.

        This ensures that dataset.read("some_id") only fails if that ID
        doesn't exist in ANY folder, not just some.
        """
        # Collect FOLDER samples
        folder_samples = [s for s in self.samples if s.type == "FOLDER"]

        # Skip validation if no folders or only one folder
        if len(folder_samples) <= 1:
            return

        # Extract NON-PADDING child ID sets from each FOLDER
        id_sets: list[tuple[str, frozenset[str]]] = []

        for folder in folder_samples:
            if hasattr(folder.path, "samples") and folder.path.samples:
                real_ids = frozenset(
                    sample.id
                    for sample in folder.path.samples
                    if not sample.id.startswith(PADDING_PREFIX)
                )
                id_sets.append((folder.id, real_ids))

        # If only one folder has children, skip
        if len(id_sets) <= 1:
            return

        # Find the folder with maximum IDs (the superset)
        max_folder_id, max_id_set = max(id_sets, key=lambda x: len(x[1]))

        # Validate: ALL other folders must be SUBSETS of max_id_set
        for folder_id, child_set in id_sets:
            if folder_id == max_folder_id:
                continue

            if not child_set.issubset(max_id_set):
                # Found IDs that are NOT in the maximum set
                extra_ids = child_set - max_id_set

                # Build error message
                error_parts = [
                    "Inclusive ID Rule violated:",
                    "The folder with maximum IDs must contain ALL IDs from other folders.",
                    "(IDs starting with '__TACOPAD__' are ignored in validation)",
                    "",
                    f"Maximum ID folder: '{max_folder_id}'",
                    f"  Non-padding IDs ({len(max_id_set)}): {sorted(max_id_set)}",
                    "",
                    f"Problematic folder: '{folder_id}'",
                    f"  Non-padding IDs ({len(child_set)}): {sorted(child_set)}",
                    "",
                    f"IDs in '{folder_id}' NOT in '{max_folder_id}': {sorted(extra_ids)}",
                    "",
                    "How to fix:",
                    "1. Ensure the folder with most files contains ALL possible IDs",
                    "2. Other folders can have fewer files (will be padded automatically)",
                    "3. Use Tortilla(samples, pad_to=N) to fill missing slots with '__TACOPAD__'",
                    "",
                    "Valid example:",
                    "  Folder A: ['cloudmask', 'data', 'thumbnail']  # Maximum",
                    "  Folder B: ['cloudmask', 'data']  # Subset of A, OK",
                    "",
                    "Invalid example:",
                    "  Folder A: ['cloudmask', 'data']",
                    "  Folder B: ['cloudmask', 'thumbnail']  # 'thumbnail' not in A, FAIL",
                ]

                raise ValueError("\n".join(error_parts))

    def _calculate_current_depth(self) -> int:
        """Calculate current depth by examining samples."""
        if not self.samples:
            return 0

        max_depth = 0

        for sample in self.samples:
            if sample.type == "FOLDER":
                tortilla_path = cast(Tortilla, sample.path)
                child_depth = 1 + tortilla_path._current_depth
                max_depth = max(max_depth, child_depth)

        return max_depth

    def __len__(self) -> int:
        """Return number of samples in this tortilla."""
        return len(self.samples)

    def extend_with(self, extension: TortillaExtension) -> "Tortilla":
        """
        Add extension data via DataFrame processing.

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
            deep: Expansion level (0-{SHARED_MAX_DEPTH} max)
                - 0: Current level only (with extensions)
                - >0: Expand N levels deep (base metadata only, adds internal:parent_id)

        Raises:
            ValueError: If deep is negative or exceeds maximum depth
        """
        # Use centralized validation from _constants.py
        validate_depth(deep, context="export_metadata")

        if deep == 0:
            return self._metadata_df.clone()

        return self._expand_hierarchical(deep)

    def _expand_hierarchical(self, target_deep: int) -> pl.DataFrame:
        """
        Build expanded DataFrame for hierarchical samples.

        Adds internal:parent_id column to link children to their parent's GLOBAL index.
        This column is permanent and enables relational queries.

        Uses cumulative global index tracking to ensure parent_id values
        reference the correct row in the previous level's DataFrame.
        """
        if target_deep > self._current_depth:
            raise ValueError(
                f"Cannot expand to depth {target_deep}: structure only has "
                f"{self._current_depth} levels (0-{self._current_depth})."
            )

        current_samples = self.samples
        current_dfs = []

        for _level in range(1, target_deep + 1):
            next_dfs = []
            next_samples = []

            # Track cumulative global index across all parents
            # Ensures parent_id references the correct row in consolidated DataFrame
            global_parent_idx = 0

            for sample in current_samples:
                if sample.type == "FOLDER" and sample.path.samples:
                    # Sample has children - process them
                    for child_sample in sample.path.samples:
                        child_metadata_df = child_sample.export_metadata()

                        # Use Int64 for parent_id consistency across ALL levels
                        # This ensures level0 (Int64) and level1+ (Int64) match for JOINs
                        child_metadata_df = child_metadata_df.with_columns(
                            pl.lit(global_parent_idx, dtype=pl.Int64).alias(
                                METADATA_PARENT_ID
                            )
                        )

                        next_dfs.append(child_metadata_df)
                        next_samples.append(child_sample)

                # Increment for every parent sample (FILE or FOLDER)
                # Ensures parent_id mapping stays aligned with parent DataFrame indices
                global_parent_idx += 1

            current_dfs = next_dfs
            current_samples = next_samples

        # Align schemas before concatenating using DRY helper function
        if len(current_dfs) > 1:
            current_dfs = align_dataframe_schemas(
                current_dfs, core_fields=["id", "type", "path", METADATA_PARENT_ID]
            )

        return pl.concat(current_dfs, how="vertical")
