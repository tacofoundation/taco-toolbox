"""
Exception classes for tacotoolbox operations.

All tacotoolbox exceptions inherit from TacoToolboxError for easy catching.
Provides specific exception types for different failure modes: validation errors,
creation failures, I/O problems, schema mismatches, consolidation issues, and
documentation generation errors.

Usage:
    from tacotoolbox._exceptions import TacoCreationError, TacoValidationError
    try:
        tacotoolbox.create(taco, "output.tacozip")
    except TacoValidationError:
        logger.warning("Invalid input parameters")
    except TacoCreationError:
        logger.error("Creation failed")
"""


class TacoToolboxError(Exception):
    """Base exception for all tacotoolbox errors."""

    pass


class TacoValidationError(TacoToolboxError):
    """
    Input validation failed before operation started.

    Raised when:
    - Output path already exists
    - Invalid parameter values (split_size, format, group_by)
    - Incompatible parameter combinations

    Examples:
        - "Output file already exists: data.tacozip"
        - "Invalid split_size format: '4XB'. Use format like '4GB'"
        - "split_size not supported with format='folder'"
        - "Group column 'region' not found in metadata"

    When caught:
        User should fix inputs and retry. No cleanup needed as no work was started.
    """

    pass


class TacoCreationError(TacoToolboxError):
    """
    Creation, translation, or export operation failed AFTER successful object construction.

    NOTE: Input validation happens BEFORE this stage via Pydantic/ValueError.
    You cannot reach TacoCreationError with empty/malformed Taco objects.
    Those fail immediately at construction time with ValueError or ValidationError.

    Raised when:
    - File I/O fails during container creation (permissions, disk full, network timeout)
    - Sample data cannot be read or copied to container
    - Container writing fails (ZIP compression, FOLDER structure)
    - Metadata generation fails during write operations
    - Temp file operations fail during container assembly

    Examples:
        - "Failed to create ZIP at 'output.tacozip': Permission denied"
        - "Failed to create FOLDER at 'output/': Disk full"
        - "Failed to convert ZIP→FOLDER: 'input.tacozip' corrupted"
        - "Failed to export to 'filtered.tacozip': Download timeout"
        - "Failed to copy sample data to container: File not readable"

    When caught:
        Operation may have created partial files. Consider cleanup.
        Check disk space, permissions, network connectivity.
    """

    pass


class TacoConsolidationError(TacoToolboxError):
    """
    Dataset consolidation failed (TacoCat or TacoCollection).

    Raised when:
    - Input datasets cannot be read or are corrupted
    - COLLECTION.json is missing or malformed
    - Metadata consolidation fails
    - PIT schema summing fails
    - Output file writing fails

    Examples:
        - "Invalid TACO file: Cannot read header from 'dataset.tacozip'"
        - "Missing 'taco:pit_schema' in dataset.tacozip"
        - "Failed to consolidate 100 datasets into TacoCat"
        - "Failed to sum PIT schemas: Sum is zero"

    When caught:
        Verify all input datasets are valid and readable.
        Check schema compatibility with validate_schema=True.
    """

    pass


class TacoSchemaError(TacoToolboxError):
    """
    Schema incompatibility detected during consolidation or alignment.

    Raised when:
    - PIT schemas differ (structure, types, hierarchy depth)
    - Field schemas differ (columns, types)
    - Required metadata columns missing
    - Schema alignment fails during consolidation

    Examples:
        - "Schema mismatch: Dataset 2 has different root type"
        - "Schema mismatch: Collection 3 has different hierarchy depth"
        - "Missing required column 'internal:gdal_vsi'"
        - "Cannot align schemas: Incompatible column types"

    When caught:
        Schemas are fundamentally incompatible.
        For consolidation, check with validate_schema=False to force (risky).
        For creation, verify all samples have compatible extensions.
    """

    pass


class TacoIOError(TacoToolboxError):
    """
    File or network I/O operation failed.

    Raised when:
    - File not found or not readable
    - Permission denied (filesystem or cloud storage)
    - Network timeout or connection error
    - HTTP errors (403/404/500)
    - S3/GCS/Azure authentication failures
    - Disk full or quota exceeded

    Examples:
        - "File not found: /path/to/input.tacozip"
        - "Permission denied writing to: /protected/output/"
        - "Failed to download from s3://bucket/data.tacozip: 403 Forbidden"
        - "Network timeout reading from https://..."
        - "Disk full: Cannot write to /data/"

    When caught:
        Check file paths, permissions, network connectivity.
        For cloud storage, verify credentials and bucket access.
        For disk full, free up space or choose different location.
    """

    pass


class TacoDocumentationError(TacoToolboxError):
    """
    Documentation generation failed.

    Raised when:
    - Required dependencies missing (jinja2, markdown)
    - TACOLLECTION.json is invalid or missing required fields
    - Template rendering fails
    - Asset files (CSS, JS, vendor libs) cannot be read
    - Output file writing fails

    Examples:
        - "Documentation requires jinja2 and markdown. Install with: pip install jinja2 markdown"
        - "Invalid JSON in TACOLLECTION.json"
        - "Failed to render HTML template: Syntax error"
        - "Failed to load asset: vendor/d3.min.js"

    When caught:
        Install missing dependencies: pip install jinja2 markdown
        Verify TACOLLECTION.json is valid and complete.
    """

    pass
