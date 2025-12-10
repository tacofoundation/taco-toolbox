"""
Tortilla datamodel for TACO framework.

Tortilla is a container for Sample collections with hierarchical metadata export.
Supports nested sample structures with position tracking.

Key features:
- Hierarchical metadata export (up to SHARED_MAX_DEPTH levels)
- Auto-padding for homogeneous structures
- Arrow Table construction with schema validation
- Configurable schema strictness (strict_schema parameter)
- Private _total_size attribute (sum of all sample sizes)

Best Practice: When mixing FOLDERs and FILEs in the same Tortilla,
place all FOLDERs before FILEs for better organization and readability.
"""

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pydantic

from tacotoolbox._column_utils import align_arrow_schemas
from tacotoolbox._constants import METADATA_PARENT_ID, SHARED_MAX_DEPTH

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample


class TortillaExtension(ABC, pydantic.BaseModel):
    """Abstract base class for Tortilla extensions that add computed columns."""

    schema_only: bool = pydantic.Field(
        False,
        description="If True, return None values while preserving schema",
        validation_alias="return_none",
    )

    @abstractmethod
    def get_schema(self) -> pa.Schema:
        """Return the expected Arrow schema for this extension."""
        pass

    @abstractmethod
    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        pass

    @abstractmethod
    def _compute(self, tortilla: "Tortilla") -> pa.Table:
        """Actual computation logic - only called when schema_only=False."""
        pass

    def __call__(self, tortilla: "Tortilla") -> pa.Table:
        """Process Tortilla and return computed metadata as Arrow Table."""
        if self.schema_only:
            arrow_schema = self.get_schema()
            none_data = {field.name: [None] for field in arrow_schema}
            return pa.Table.from_pydict(none_data, schema=arrow_schema)

        return self._compute(tortilla)


class Tortilla:
    """
    Container for Sample collections with hierarchical metadata export.

    Supports nested sample structures with position tracking (max depth from constants).

    Best Practice: When mixing FOLDERs and FILEs in the same Tortilla,
    place all FOLDERs before FILEs for better organization and readability.

    Auto-Padding: Use `pad_to` parameter to automatically pad sample list to
    make length divisible by a specific value. Useful for creating homogeneous
    tree structures. Padding samples have IDs starting with "__TACOPAD__" and
    preserve the schema of real samples with None values.

    Schema Validation: By default (strict_schema=True), all samples must have
    identical metadata schemas. Set strict_schema=False to allow heterogeneous
    schemas with automatic None-filling for missing columns.

    Private Attributes:
    - _current_depth: Maximum depth of nested structure
    - _size_bytes: Sum of all sample sizes in bytes (same name as Sample)
    - _field_descriptions: Field descriptions from extensions
    """

    def __init__(
        self,
        samples: list["Sample"],
        pad_to: int | None = None,
        strict_schema: bool = True,
    ) -> None:
        """
        Create Tortilla with Arrow Table construction and schema validation.

        Raises:
            ValueError: If samples have inconsistent metadata schemas (strict_schema=True)
                       or empty list or duplicate IDs
        """
        if not samples:
            raise ValueError("Cannot create Tortilla with empty samples list")

        # Auto-pad if requested
        if pad_to is not None:
            samples = self._create_padded_samples(samples, pad_to)

        self.samples = samples

        # Initialize field descriptions tracker
        self._field_descriptions: dict[str, str] = {}

        # Validate unique IDs at this level
        self._validate_unique_ids()

        # Check ordering best practice (only for mixed types)
        self._check_sample_ordering()

        # Extract metadata from all samples
        metadata_tables = []
        for sample in samples:
            metadata_tables.append(sample.export_metadata())

            # Collect field descriptions from samples
            if hasattr(sample, "_field_descriptions"):
                self._field_descriptions.update(sample._field_descriptions)

        # Handle schema validation/alignment
        if strict_schema:
            reference_schema = metadata_tables[0].schema
            for i, table in enumerate(metadata_tables[1:], start=1):
                if not table.schema.equals(reference_schema):
                    self._raise_schema_mismatch_error(
                        i, samples[i], reference_schema, table.schema
                    )
        else:
            metadata_tables = align_arrow_schemas(
                metadata_tables, core_fields=["id", "type", "path"]
            )

        # Concatenate Arrow Tables
        self._metadata_table = pa.concat_tables(metadata_tables)

        # Calculate depth and size
        self._current_depth = self._calculate_current_depth()
        self._size_bytes = sum(s._size_bytes for s in self.samples)

    @property
    def metadata_table(self) -> pa.Table:
        """Metadata table for extension computation."""
        return self._metadata_table

    @staticmethod
    def _validate_depth(depth: int, context: str = "operation") -> None:
        """
        Validate that depth is within allowed range.

        Args:
            depth: Depth value to validate
            context: Context string for error message

        Raises:
            ValueError: If depth is invalid
        """
        if depth < 0:
            raise ValueError(f"{context}: depth must be non-negative, got {depth}")

        if depth > SHARED_MAX_DEPTH:
            raise ValueError(
                f"{context}: depth {depth} exceeds maximum of {SHARED_MAX_DEPTH} "
                f"(levels 0-{SHARED_MAX_DEPTH})"
            )

    def _raise_schema_mismatch_error(
        self,
        sample_index: int,
        sample: "Sample",
        reference_schema: pa.Schema,
        current_schema: pa.Schema,
    ) -> None:
        """Raise helpful error when schema mismatch detected in strict mode."""
        reference_columns = set(reference_schema.names)
        current_columns = set(current_schema.names)

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
            "  Hint: If you want to combine samples with different schemas, use:\n"
            "      Tortilla(samples=[...], strict_schema=False)\n\n"
            "  This will auto-fill missing columns with None values.\n\n"
            "  To inspect sample schemas before creating Tortilla:\n"
            "      table = sample.export_metadata()\n"
            "      print(table.schema)  # See schema structure\n"
        )

        raise ValueError(error_msg)

    def _validate_unique_ids(self) -> None:
        """
        Ensure all sample IDs are unique at this level.

        Duplicate IDs cause silent failures in ZIP offset calculation because
        offsets are stored in a dictionary keyed by archive path. When duplicate
        IDs exist, later samples overwrite earlier ones in the offset map,
        resulting in missing files in the final container.
        """
        ids = [s.id for s in self.samples]
        duplicates = {
            sample_id: count for sample_id, count in Counter(ids).items() if count > 1
        }

        if duplicates:
            dup_list = ", ".join(
                f"'{sample_id}' ({count}x)"
                for sample_id, count in list(duplicates.items())[:10]
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
        ref_table = samples[0].export_metadata()

        padded_samples = samples.copy()

        for i in range(num_padding):
            # Create padding sample using internal factory
            # This creates a 0-byte temp file that bypasses ID validation
            dummy = Sample._create_padding(index=i)

            # Get extension columns (excluding core fields)
            extension_cols = [
                field.name
                for field in ref_table.schema
                if field.name not in ["id", "type", "path"]
            ]

            if extension_cols:
                # Create Arrow Table with None values using reference schema
                extension_schema = pa.schema(
                    [ref_table.schema.field(col) for col in extension_cols]
                )
                none_data = {col: [None] for col in extension_cols}
                padding_table = pa.Table.from_pydict(none_data, schema=extension_schema)

                # Extend dummy with padding table
                dummy.extend_with(padding_table)

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

        if (
            first_file_idx is not None
            and first_folder_idx is not None
            and first_file_idx < first_folder_idx
        ):
            warnings.warn(
                f"Consider placing FOLDERs before FILEs for better organization. "
                f"Found FILE at position {first_file_idx} before FOLDER at position {first_folder_idx}.",
                UserWarning,
                stacklevel=3,
            )

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
        Add extension data via Arrow Table processing.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If row count mismatch or column conflicts
        """
        result_table = extension(self)

        if result_table.num_rows != self._metadata_table.num_rows:
            raise ValueError(
                f"Extension returned {result_table.num_rows} rows, "
                f"expected {self._metadata_table.num_rows} (one per sample)."
            )

        conflicts = set(result_table.schema.names) & set(
            self._metadata_table.schema.names
        )
        if conflicts:
            raise ValueError(f"Column conflicts: {sorted(conflicts)}")

        # Capture field descriptions if extension provides them
        if hasattr(extension, "get_field_descriptions"):
            descriptions = extension.get_field_descriptions()
            self._field_descriptions.update(descriptions)

        combined_schema = pa.schema(
            list(self._metadata_table.schema) + list(result_table.schema)
        )

        combined_arrays = [
            self._metadata_table.column(i)
            for i in range(self._metadata_table.num_columns)
        ]
        combined_arrays.extend(
            [result_table.column(i) for i in range(result_table.num_columns)]
        )

        self._metadata_table = pa.Table.from_arrays(
            combined_arrays, schema=combined_schema
        )
        return self

    def export_metadata(self, deep: int = 0) -> pa.Table:
        """
        Export metadata with optional hierarchical expansion.

        Args:
            deep: Expansion level (0-max depth)
                - 0: Current level only (with extensions)
                - >0: Expand N levels deep (base metadata only, adds internal:parent_id)

        Raises:
            ValueError: If deep is negative or exceeds maximum depth
        """
        Tortilla._validate_depth(deep, context="export_metadata")

        if deep == 0:
            return pa.Table.from_arrays(
                [
                    self._metadata_table.column(i)
                    for i in range(self._metadata_table.num_columns)
                ],
                schema=self._metadata_table.schema,
            )

        return self._expand_hierarchical(deep)

    def _expand_hierarchical(self, target_deep: int) -> pa.Table:
        """
        Build expanded Arrow Table for hierarchical samples.

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
        current_tables = []

        for _level in range(1, target_deep + 1):
            next_tables = []
            next_samples = []

            for global_parent_idx, sample in enumerate(current_samples):
                if sample.type == "FOLDER":
                    # Cast to Tortilla to ensure mypy knows it has .samples
                    tortilla_path = cast(Tortilla, sample.path)

                    if tortilla_path.samples:
                        # Sample has children - process them
                        for child_sample in tortilla_path.samples:
                            child_metadata_table = child_sample.export_metadata()

                            parent_id_array = pa.array(
                                [global_parent_idx], type=pa.int64()
                            )
                            parent_id_field = pa.field(METADATA_PARENT_ID, pa.int64())

                            child_metadata_table = child_metadata_table.append_column(
                                parent_id_field, parent_id_array
                            )

                            next_tables.append(child_metadata_table)
                            next_samples.append(child_sample)

            current_tables = next_tables
            current_samples = next_samples

        # Align schemas before concatenating
        if len(current_tables) > 1:
            current_tables = align_arrow_schemas(
                current_tables, core_fields=["id", "type", "path", METADATA_PARENT_ID]
            )

        return pa.concat_tables(current_tables)
