import contextlib
import json
import pathlib
import tempfile
import uuid
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import tacozip

from tacotoolbox._metadata import MetadataPackage, add_offset_columns
from tacotoolbox._virtual_zip import VirtualTACOZIP


class ZipWriterError(Exception):
    """Raised when ZIP writing operations fail."""


class ZipWriter:
    """Handle creation of .tacozip container files with precalculated offsets."""
    
    def __init__(
        self,
        output_path: pathlib.Path,
        quiet: bool = False,
        temp_dir: pathlib.Path | None = None
    ) -> None:
        self.output_path = output_path
        self.quiet = quiet
        
        if temp_dir is None:
            self.temp_dir = pathlib.Path(tempfile.gettempdir())
        else:
            self.temp_dir = pathlib.Path(temp_dir)
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_files: list[pathlib.Path] = []
    
    def create_complete_zip(
        self,
        src_files: list[str],
        arc_files: list[str],
        metadata_package: MetadataPackage,
        **parquet_kwargs: Any,
    ) -> pathlib.Path:
        try:
            if not self.quiet:
                print("="*60)
                print("STARTING BOTTOM-UP __meta__ GENERATION")
                print("="*60)
            
            # STEP 1: Setup VirtualZIP with data files only
            if not self.quiet:
                print("\n[STEP 1] Adding data files to VirtualZIP...")
            
            virtual_zip = VirtualTACOZIP()
            num_entries = len(metadata_package.levels) + 1
            virtual_zip.add_header(num_entries=num_entries)
            
            for src_path, arc_path in zip(src_files, arc_files):
                if pathlib.Path(src_path).exists():
                    virtual_zip.add_file(src_path, arc_path)
            
            if not self.quiet:
                print(f"  Added {len(src_files)} data files")
            
            # STEP 2: Calculate initial offsets (FILES only)
            if not self.quiet:
                print("\n[STEP 2] Calculating initial offsets for data files...")
            
            virtual_zip.calculate_offsets()
            offsets_map = virtual_zip.get_all_offsets()
            
            if not self.quiet:
                print(f"  Initial offsets calculated: {len(offsets_map)} entries")
            
            # STEP 3: Extract and group folders by depth
            if not self.quiet:
                print("\n[STEP 3] Analyzing folder hierarchy...")
            
            folder_order = self._extract_folder_order(arc_files)
            folders_by_depth = self._group_folders_by_depth(folder_order)
            
            if not self.quiet:
                for depth in sorted(folders_by_depth.keys(), reverse=True):
                    print(f"  Depth {depth}: {len(folders_by_depth[depth])} folders")
            
            # STEP 4: Bottom-up __meta__ generation
            if not self.quiet:
                print("\n[STEP 4] Generating __meta__ files (bottom-up)...")
            
            temp_csv_files = {}
            enriched_metadata_by_depth = {}
            max_depth = max(folders_by_depth.keys()) if folders_by_depth else 0
            
            for depth in range(max_depth, 0, -1):
                if not self.quiet:
                    print(f"\n  Processing depth {depth}...")
                
                folders_at_depth = folders_by_depth[depth]
                enriched_metadata_by_depth[depth] = []
                
                for folder_path in folders_at_depth:
                    if folder_path not in metadata_package.local_metadata:
                        if not self.quiet:
                            print(f"    WARNING: {folder_path} not in local_metadata, skipping")
                        continue
                    
                    # Get local metadata for this folder
                    local_df = metadata_package.local_metadata[folder_path]
                    
                    # Enrich with current offsets
                    enriched_df = add_offset_columns(local_df, offsets_map, folder_path)
                    
                    # Store enriched DataFrame for consolidated metadata rebuild
                    enriched_metadata_by_depth[depth].append(enriched_df)
                    
                    # Write CSV to temp
                    meta_arc_path = f"{folder_path}__meta__"
                    temp_csv = self._write_single_csv(enriched_df, folder_path, meta_arc_path)
                    temp_csv_files[meta_arc_path] = temp_csv
                    
                    # Add to VirtualZIP
                    real_size = temp_csv.stat().st_size
                    virtual_zip.add_file(str(temp_csv), meta_arc_path, file_size=real_size)
                    
                    if not self.quiet:
                        print(f"    Created {meta_arc_path} ({real_size} bytes)")
                
                # Recalculate offsets after processing this depth
                virtual_zip.calculate_offsets()
                offsets_map = virtual_zip.get_all_offsets()
                
                if not self.quiet:
                    print(f"  Recalculated offsets: {len(offsets_map)} entries")
            
            # STEP 5: Rebuild consolidated metadata from enriched local metadata
            if not self.quiet:
                print("\n[STEP 5] Rebuilding consolidated metadata (METADATA/levelX.parquet)...")
            
            # Level 0: Enrich with final offsets (FOLDERs now point to __meta__)
            metadata_package.levels[0] = add_offset_columns(
                metadata_package.levels[0],
                offsets_map,
                folder_path=""
            )
            
            if not self.quiet:
                print(f"  Level 0: {len(metadata_package.levels[0])} samples")
            
            # Level 1+: Concatenate enriched local metadata and sort by parent_id
            for depth in range(1, len(metadata_package.levels)):
                if depth in enriched_metadata_by_depth and enriched_metadata_by_depth[depth]:
                    concatenated = pl.concat(
                        enriched_metadata_by_depth[depth],
                        how="vertical"
                    )
                    
                    if "internal:parent_id" in concatenated.columns:
                        concatenated = concatenated.sort("internal:parent_id", maintain_order=True)
                    
                    metadata_package.levels[depth] = concatenated
                    
                    if not self.quiet:
                        print(f"  Level {depth}: {len(concatenated)} samples (rebuilt from {len(enriched_metadata_by_depth[depth])} folders)")
                else:
                    if not self.quiet:
                        print(f"  Level {depth}: No data (skipped)")
            
            # Write consolidated metadata to temp parquet files
            temp_parquet_files = {}
            for i, level_df in enumerate(metadata_package.levels):
                arc_path = f"METADATA/level{i}.parquet"
                temp_path = self.temp_dir / f"{uuid.uuid4().hex}_level{i}.parquet"
                arrow_table = level_df.to_arrow()
                pq.write_table(arrow_table, temp_path, **parquet_kwargs)
                real_size = temp_path.stat().st_size
                virtual_zip.add_file(str(temp_path), arc_path, file_size=real_size)
                temp_parquet_files[arc_path] = temp_path
                self._temp_files.append(temp_path)
                
                if not self.quiet:
                    print(f"  Added {arc_path} ({real_size} bytes)")
            
            # STEP 6: Add COLLECTION.json
            if not self.quiet:
                print("\n[STEP 6] Adding COLLECTION.json...")
            
            collection = metadata_package.collection.copy()
            collection["taco:pit_schema"] = metadata_package.pit_schema
            temp_json = self.temp_dir / f"{uuid.uuid4().hex}.json"
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(collection, f, indent=4, ensure_ascii=False)
            collection_size = temp_json.stat().st_size
            virtual_zip.add_file(str(temp_json), "COLLECTION.json", file_size=collection_size)
            self._temp_files.append(temp_json)
            
            if not self.quiet:
                print(f"  Added COLLECTION.json ({collection_size} bytes)")
            
            # STEP 7: Final offset calculation
            if not self.quiet:
                print("\n[STEP 7] Final offset calculation...")
            
            virtual_zip.calculate_offsets()
            
            # STEP 8: Build final file lists
            if not self.quiet:
                print("\n[STEP 8] Preparing final file lists for ZIP creation...")
            
            all_src_files = list(src_files)
            all_arc_files = list(arc_files)
            
            # Add __meta__ files in depth order
            for depth in range(max_depth, 0, -1):
                for folder_path in folders_by_depth[depth]:
                    meta_arc_path = f"{folder_path}__meta__"
                    if meta_arc_path in temp_csv_files:
                        temp_path = temp_csv_files[meta_arc_path]
                        all_src_files.append(str(temp_path))
                        all_arc_files.append(meta_arc_path)
            
            # Add parquet files
            for i in range(len(metadata_package.levels)):
                arc_path = f"METADATA/level{i}.parquet"
                temp_path = temp_parquet_files[arc_path]
                all_src_files.append(str(temp_path))
                all_arc_files.append(arc_path)
            
            # Add COLLECTION.json
            all_src_files.append(str(temp_json))
            all_arc_files.append("COLLECTION.json")
            
            if not self.quiet:
                print(f"  Total files in ZIP: {len(all_src_files)}")
            
            # STEP 9: Write final ZIP
            if not self.quiet:
                print("\n[STEP 9] Writing final ZIP file...")
            
            header_entries = [(0, 0) for _ in range(num_entries)]
            
            tacozip.create(
                zip_path=str(self.output_path),
                src_files=all_src_files,
                arc_files=all_arc_files,
                entries=header_entries
            )
            
            if not self.quiet:
                print(f"\n{'='*60}")
                print(f"ZIP CREATED SUCCESSFULLY: {self.output_path}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            raise ZipWriterError(f"Failed to create ZIP: {e}") from e
        else:
            return self.output_path
        finally:
            self._cleanup()
    
    def _extract_folder_order(self, arc_files: list[str]) -> list[str]:
        """
        Extract folder paths that need __meta__ files.
        
        Only includes Level 1+ folders (not DATA/ root).
        """
        folder_set = set()
        
        for arc_path in arc_files:
            if "/" in arc_path:
                parts = arc_path.split("/")
                for i in range(2, len(parts)):
                    folder_path = "/".join(parts[:i]) + "/"
                    folder_set.add(folder_path)
        
        return sorted(folder_set)
    
    def _group_folders_by_depth(self, folder_order: list[str]) -> dict[int, list[str]]:
        """
        Group folders by depth for bottom-up processing.
        
        Returns:
            Dictionary mapping depth -> list of folder paths
            Example: {3: ["DATA/a/b/c/"], 2: ["DATA/a/b/"], 1: ["DATA/a/"]}
        """
        by_depth: dict[int, list[str]] = {}
        
        for folder in folder_order:
            # Calculate depth: "DATA/big1/" -> depth 1, "DATA/big1/s1/" -> depth 2
            depth = folder.count("/") - 1
            
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(folder)
        
        return by_depth
    
    def _write_single_csv(
        self,
        df: pl.DataFrame,
        folder_path: str,
        meta_arc_path: str
    ) -> pathlib.Path:
        """Write a single __meta__ CSV file to temp directory."""
        identifier = folder_path.replace("/", "_").strip("_")
        temp_path = self.temp_dir / f"{uuid.uuid4().hex}_{identifier}.csv"
        
        filtered_df = self._filter_csv_columns(df)
        filtered_df.write_csv(temp_path)
        
        self._temp_files.append(temp_path)
        return temp_path
    
    def _filter_csv_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter to only required columns for __meta__ CSV."""
        required = [
            "id",
            "type",
            "internal:offset",
            "internal:size"
        ]
        
        available = [col for col in required if col in df.columns]
        
        return df.select(available)
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            with contextlib.suppress(Exception):
                temp_file.unlink(missing_ok=True)
        self._temp_files.clear()