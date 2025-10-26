import json
import pathlib
import shutil
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from tacotoolbox._metadata import MetadataPackage


class FolderWriterError(Exception):
    """Raised when folder writing operations fail."""


class FolderWriter:
    """Handle creation of folder container structures with dual metadata."""
    
    def __init__(self, output_dir: pathlib.Path, quiet: bool = True) -> None:
        self.output_dir = output_dir
        self.quiet = quiet
        self.data_dir = output_dir / "DATA"
        self.metadata_dir = output_dir / "METADATA"
    
    def create_complete_folder(
        self,
        samples: list[Any],
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> pathlib.Path:
        try:
            if not self.quiet:
                print("Creating folder structure...")
            
            self._create_structure()
            
            if not self.quiet:
                print("Copying data files...")
            self._copy_data_files(samples)
            
            if not self.quiet:
                print("Writing consolidated metadata...")
            self._write_consolidated_metadata(metadata_package, **kwargs)
            
            if not self.quiet:
                print("Writing COLLECTION.json...")
            self._write_collection_json(
                metadata_package.collection,
                metadata_package.pit_schema
            )
            
            if not self.quiet:
                print(f"Folder container created: {self.output_dir}/")
        
        except Exception as e:
            raise FolderWriterError(f"Failed to create folder container: {e}") from e
        else:
            return self.output_dir
    
    def _create_structure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=False)
        self.metadata_dir.mkdir(parents=True, exist_ok=False)
    
    def _copy_data_files(self, samples: list[Any]) -> None:
        self._copy_samples_recursive(samples, path_prefix="")
    
    def _copy_samples_recursive(
        self,
        samples: list[Any],
        path_prefix: str = ""
    ) -> None:
        for sample in samples:
            if sample.type == "FOLDER":
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                nested_dir = self.data_dir / new_prefix.rstrip("/")
                nested_dir.mkdir(parents=True, exist_ok=True)
                
                self._copy_samples_recursive(
                    sample.path.samples, path_prefix=new_prefix
                )
            else:
                src_path = sample.path
                # Use sample.id as-is for destination filename
                dst_path = self.data_dir / f"{path_prefix}{sample.id}"
                
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(src_path, dst_path)
    
    def _write_consolidated_metadata(
        self,
        metadata_package: MetadataPackage,
        **parquet_kwargs: Any,
    ) -> None:
        for i, level_df in enumerate(metadata_package.levels):
            output_path = self.metadata_dir / f"level{i}.parquet"
            arrow_table = level_df.to_arrow()
            pq.write_table(arrow_table, output_path, **parquet_kwargs)
    
    def _write_collection_json(
        self,
        collection: dict[str, object],
        pit_schema: dict
    ) -> None:
        collection_path = self.output_dir / "COLLECTION.json"
        
        collection_with_schema = collection.copy()
        collection_with_schema["taco:pit_schema"] = pit_schema
        
        with open(collection_path, "w", encoding="utf-8") as f:
            json.dump(collection_with_schema, f, indent=4, ensure_ascii=False)