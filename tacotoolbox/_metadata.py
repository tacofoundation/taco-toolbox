import pathlib
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from tacotoolbox.taco.datamodel import Taco
    from tacotoolbox.sample.datamodel import Sample


class PITValidationError(Exception):
    """Raised when Position-Isomorphic Tree constraint is violated."""


class MetadataPackage:
    """
    Complete metadata bundle with dual system.
    
    Attributes:
        levels: Consolidated metadata for METADATA/levelX.parquet (all levels 0-5)
        local_metadata: Local metadata for DATA/folder/__metadata__ (FOLDERs only, level 1+)
        collection: COLLECTION.json content
        pit_schema: PIT schema for navigation
        max_depth: Maximum hierarchy depth (max 5, meaning 6 levels: 0-5)
    """
    
    def __init__(
        self,
        levels: list[pl.DataFrame],
        local_metadata: dict[str, pl.DataFrame],
        collection: dict[str, Any],
        pit_schema: dict[str, Any],
        max_depth: int
    ):
        self.levels = levels
        self.local_metadata = local_metadata
        self.collection = collection
        self.pit_schema = pit_schema
        self.max_depth = max_depth


class MetadataGenerator:
    """Generate dual metadata system: consolidated + local."""
    
    def __init__(self, taco: "Taco", quiet: bool = False) -> None:
        self.taco = taco
        self.quiet = quiet
        self.max_depth = min(taco.tortilla._current_depth, 5)
    
    def generate_all_levels(self) -> MetadataPackage:
        levels = []
        dataframes = []
        
        for depth in range(self.max_depth + 1):
            df = self.taco.tortilla.export_metadata(deep=depth)
            df = self._clean_dataframe(df)
            dataframes.append(df)
            
            if depth == 0:
                self._validate_pit_level0(df)
            else:
                self._validate_pit_depth(df, dataframes[depth - 1], depth)
        
        pit_schema = generate_pit_schema(dataframes)
        
        # Keep internal:parent_id column for relational queries
        levels = dataframes
        
        local_metadata = {}
        
        for sample in self.taco.tortilla.samples:
            if sample.type == "FOLDER":
                folder_path = f"DATA/{sample.id}/"
                folder_df = self._generate_folder_metadata(sample)
                local_metadata[folder_path] = folder_df
                
                nested = self._generate_nested_folders(sample, f"DATA/{sample.id}/")
                local_metadata.update(nested)
        
        collection = generate_collection_json(self.taco)
        
        return MetadataPackage(
            levels=levels,
            local_metadata=local_metadata,
            collection=collection,
            pit_schema=pit_schema,
            max_depth=self.max_depth
        )
    
    def _generate_folder_metadata(self, folder_sample: "Sample") -> pl.DataFrame:
        samples = folder_sample.path.samples
        metadata_dfs = [s.export_metadata() for s in samples]
        df = pl.concat(metadata_dfs, how="vertical")
        return self._clean_dataframe(df)
    
    def _generate_nested_folders(
        self,
        parent_sample: "Sample",
        parent_path: str
    ) -> dict[str, pl.DataFrame]:
        result = {}
        
        for child in parent_sample.path.samples:
            if child.type == "FOLDER":
                folder_path = f"{parent_path}{child.id}/"
                folder_df = self._generate_folder_metadata(child)
                result[folder_path] = folder_df
                
                nested = self._generate_nested_folders(child, folder_path)
                result.update(nested)
        
        return result
    
    def _validate_pit_level0(self, df: pl.DataFrame) -> None:
        if "type" not in df.columns:
            raise PITValidationError("Level 0 missing 'type' column")
        
        normalized_types = [_normalize_type(t) for t in df["type"].to_list()]
        unique_types = list(set(normalized_types))
        
        if len(unique_types) != 1:
            raise PITValidationError(
                f"PIT constraint violated at level 0:\n"
                f"All nodes must have the same type.\n"
                f"Found types: {unique_types}"
            )
    
    def _validate_pit_depth(
        self,
        df: pl.DataFrame,
        parent_df: pl.DataFrame,
        depth: int
    ) -> None:
        if "type" not in df.columns:
            raise PITValidationError(f"Depth {depth} missing 'type' column")
        
        parent_types = [_normalize_type(t) for t in parent_df["type"].to_list()]
        parent_pattern = self._infer_unique_pattern(parent_types, depth - 1)
        folder_positions = [i for i, t in enumerate(parent_pattern) if t == "FOLDER"]
        
        if not folder_positions:
            raise PITValidationError(
                f"Depth {depth} exists but no FOLDERs at depth {depth - 1}"
            )
        
        num_parents = len(parent_df)
        child_types = [_normalize_type(t) for t in df["type"].to_list()]
        
        for folder_idx, position in enumerate(folder_positions):
            chunk_pattern = self._extract_chunk_pattern(
                child_types, num_parents, len(folder_positions), folder_idx
            )
            
            if chunk_pattern is None:
                raise PITValidationError(
                    f"PIT constraint violated at depth {depth}:\n"
                    f"Cannot extract consistent pattern for FOLDER at position {position}"
                )
            
            for parent_idx in range(num_parents):
                chunk_start = (parent_idx * len(folder_positions) + folder_idx) * len(
                    chunk_pattern
                )
                chunk_end = chunk_start + len(chunk_pattern)
                actual_chunk = child_types[chunk_start:chunk_end]
                
                if actual_chunk != chunk_pattern:
                    raise PITValidationError(
                        f"PIT constraint violated at depth {depth}:\n"
                        f"FOLDER at position {position}, parent {parent_idx} has different pattern.\n"
                        f"Expected: {chunk_pattern}\n"
                        f"Actual: {actual_chunk}"
                    )
    
    def _infer_unique_pattern(self, types: list[str], depth: int) -> list[str]:
        total = len(types)
        
        for pattern_len in range(1, total // 2 + 1):
            if total % pattern_len != 0:
                continue
            
            pattern = types[:pattern_len]
            num_repeats = total // pattern_len
            
            if all(
                types[i * pattern_len : (i + 1) * pattern_len] == pattern
                for i in range(num_repeats)
            ):
                return pattern
        
        return types
    
    def _extract_chunk_pattern(
        self,
        types: list[str],
        num_parents: int,
        num_folders_per_parent: int,
        folder_idx: int,
    ) -> list[str] | None:
        total_types = len(types)
        expected_total = num_parents * num_folders_per_parent
        
        if total_types % expected_total != 0:
            return None
        
        chunk_size = total_types // expected_total
        
        first_chunk_start = folder_idx * chunk_size
        first_chunk_end = first_chunk_start + chunk_size
        pattern = types[first_chunk_start:first_chunk_end]
        
        for parent_idx in range(num_parents):
            for fld_idx in range(num_folders_per_parent):
                if fld_idx == folder_idx:
                    chunk_start = (
                        parent_idx * num_folders_per_parent + fld_idx
                    ) * chunk_size
                    chunk_end = chunk_start + chunk_size
                    if types[chunk_start:chunk_end] != pattern:
                        return None
        
        return pattern
    
    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        container_irrelevant_cols = ["path"]
        df = df.drop([col for col in container_irrelevant_cols if col in df.columns])
        
        cols_to_keep = []
        
        for col in df.columns:
            if df[col].is_null().all():
                continue
            
            if df[col].dtype == pl.Utf8:
                non_empty = df.filter(
                    (pl.col(col).is_not_null())
                    & (pl.col(col) != "")
                    & (pl.col(col) != "None")
                ).height
                if non_empty > 0:
                    cols_to_keep.append(col)
            else:
                cols_to_keep.append(col)
        
        return df.select(cols_to_keep) if cols_to_keep else df.select([df.columns[0]])


def _remove_empty_internal_columns(df: pl.DataFrame) -> pl.DataFrame:
    protected_columns = {
        "internal:offset",
        "internal:size",
        "internal:relative_path",
        "internal:parent_id",
    }
    
    cols_to_drop = []
    
    for col in df.columns:
        if (
            col.startswith("internal:")
            and col not in protected_columns
            and df[col].is_null().all()
        ):
            cols_to_drop.append(col)
    
    return df.drop(cols_to_drop) if cols_to_drop else df


def add_offset_columns(
    metadata_df: pl.DataFrame,
    offsets_map: dict[str, tuple[int, int]],
    folder_path: str = ""
) -> pl.DataFrame:
    """
    Add internal:offset and internal:size columns to metadata DataFrame.
    
    Critical behavior:
    - FILE samples: offset points to the actual data file
    - FOLDER samples: offset points to the child's __meta__ file
    
    Args:
        metadata_df: DataFrame with sample metadata
        offsets_map: Dictionary mapping arc_path -> (offset, size)
        folder_path: Current folder path (e.g., "DATA/bigsample1/")
    
    Returns:
        DataFrame with added offset columns
    """
    offsets = []
    sizes = []
    
    for row in metadata_df.iter_rows(named=True):
        sample_id = row["id"]
        sample_type = row["type"]
        
        if sample_type == "FOLDER":
            # FOLDER: offset points to child's __meta__ file
            if folder_path:
                # We're inside a folder (Level 1+)
                meta_path = f"{folder_path}{sample_id}/__meta__"
            else:
                # We're at Level 0 (METADATA/level0.parquet)
                meta_path = f"DATA/{sample_id}/__meta__"
            
            if meta_path in offsets_map:
                offset, size = offsets_map[meta_path]
                offsets.append(offset)
                sizes.append(size)
            else:
                offsets.append(None)
                sizes.append(None)
        else:
            # FILE: offset points to actual data file
            if folder_path:
                arc_path = f"{folder_path}{sample_id}"
            else:
                arc_path = f"DATA/{sample_id}"
            
            if arc_path in offsets_map:
                offset, size = offsets_map[arc_path]
                offsets.append(offset)
                sizes.append(size)
            else:
                offsets.append(None)
                sizes.append(None)
    
    result_df = metadata_df.with_columns([
        pl.Series("internal:offset", offsets),
        pl.Series("internal:size", sizes)
    ])
    
    return _remove_empty_internal_columns(result_df)


def generate_pit_schema(dataframes: list[pl.DataFrame]) -> dict[str, Any]:
    """
    Generate PIT schema with id and type arrays.
    
    For each pattern, finds the group with maximum real samples (no __TACOPAD__)
    to extract canonical id and type lists.
    """
    if not dataframes:
        raise PITValidationError("Need at least one DataFrame to generate schema")
    
    df0 = dataframes[0]
    if "type" not in df0.columns:
        raise PITValidationError("Level 0 missing 'type' column")
    
    root_type = _normalize_type(df0["type"][0])
    root = {"n": len(df0), "type": root_type}
    
    hierarchy: dict[str, list[dict]] = {}
    
    for depth in range(1, len(dataframes)):
        df = dataframes[depth]
        parent_df = dataframes[depth - 1]
        
        if len(df) == 0:
            continue
        
        if "internal:parent_id" not in df.columns:
            raise PITValidationError(
                f"Depth {depth} missing 'internal:parent_id' column"
            )
        
        if depth == 1:
            # Level 1: Simple case - just take first parent's children
            children_per_parent = len(df) // len(parent_df)
            first_parent_children = df.head(children_per_parent)
            
            # Get types and ids
            child_types = [
                _normalize_type(t) for t in first_parent_children["type"].to_list()
            ]
            child_ids = first_parent_children["id"].to_list()
            
            pattern = {"n": len(df), "type": child_types, "id": child_ids}
            hierarchy[str(depth)] = [pattern]
        
        else:
            # Level 2+: Need to handle multiple FOLDER positions
            parent_schema = hierarchy[str(depth - 1)]
            parent_pattern = parent_schema[0]["type"]
            pattern_size = len(parent_pattern)
            num_groups = len(parent_df) // pattern_size
            
            folder_positions = [
                i for i, t in enumerate(parent_pattern) if t == "FOLDER"
            ]
            
            if not folder_positions:
                continue
            
            all_patterns: list[dict] = []
            
            for position_idx in folder_positions:
                parent_ids_for_position = [
                    group_idx * pattern_size + position_idx
                    for group_idx in range(num_groups)
                ]
                
                position_children = df.filter(
                    pl.col("internal:parent_id").is_in(parent_ids_for_position)
                )
                
                if len(position_children) == 0:
                    continue
                
                # Each group represents one parent's children
                # All groups should have same structure, find the one with most real samples
                samples_per_group = len(position_children) // num_groups
                
                max_real_count = 0
                best_group_ids = []
                best_group_types = []
                
                for group_idx in range(num_groups):
                    start_idx = group_idx * samples_per_group
                    end_idx = start_idx + samples_per_group
                    group = position_children[start_idx:end_idx]
                    
                    ids = group["id"].to_list()
                    types = [_normalize_type(t) for t in group["type"].to_list()]
                    
                    real_count = sum(
                        1 for id_val in ids
                        if not str(id_val).startswith("__TACOPAD__")
                    )
                    
                    if real_count > max_real_count:
                        max_real_count = real_count
                        best_group_ids = ids
                        best_group_types = types
                
                total_nodes = num_groups * len(best_group_types)
                
                pattern_dict = {
                    "n": total_nodes,
                    "type": best_group_types,
                    "id": best_group_ids,
                }
                all_patterns.append(pattern_dict)
            
            hierarchy[str(depth)] = all_patterns
    
    return {"root": root, "hierarchy": hierarchy}


def generate_collection_json(taco: "Taco") -> dict[str, Any]:
    collection = taco.model_dump()
    collection.pop("tortilla", None)
    return collection


def _normalize_type(type_str: str) -> str:
    return type_str