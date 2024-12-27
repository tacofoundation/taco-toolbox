from typing import Union
import pathlib
import mmap
import pandas as pd
import tacotoolbox.datamodel
import pyarrow as pa
import pyarrow.parquet as pq


def edit_collection(
    taco: Union[str, pathlib.Path],
    collection: tacotoolbox.datamodel.Collection    
) -> pathlib.Path:
    """Edit the Collection of a TACO file 🌮.

    Sometimes you may want to add a new metadata to the collection bytes.

    Args:
        taco (Union[str, pathlib.Path]): The path to
            the TACO file.
        collection (tacotoolbox.datamodel.Collection): The new 
            collection of the TACO file.
            
    Returns:
        pathlib.Path: Path to the updated TACO file.
    """
    # Check if the taco file exists
    taco = pathlib.Path(taco)
    if not taco.exists():
        raise FileNotFoundError(f"The TACO file 🌮 '{taco}' does not exist.")
    
    # Convert the Collection to a dictionary
    metadata_bytes: bytes = collection.model_dump_json().encode()
    metadata_size: int = len(metadata_bytes)
    metadata_size_b: bytes = metadata_size.to_bytes(8, byteorder="little")

    # Update the offset and length
    with open(taco, "r+b") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
            # Check the magic number
            magic = mm[0:2]
            if magic != b"WX":
                raise ValueError("Invalid file type: must be a TACO 🌮")

            # Read the collection offset (MO) and length (CL)
            metadata_offset = int.from_bytes(mm[26:34], byteorder="little")

            # Update the collection length (CL)
            mm[34:42] = metadata_size_b

            # Move the pointer to the metadata offset
            mm.seek(metadata_offset)

            # Truncate the file if the new metadata is smaller or larger
            mm.resize(metadata_offset + metadata_size)
            
            # Overwrite the metadata
            mm.write(metadata_bytes)

    print(f"Collection updated successfully! 🌮")
    return taco


def edit_footer(
    taco: Union[str, pathlib.Path],
    dataframe: pd.DataFrame
) -> pathlib.Path:
    """Edit the Footer of a TACO file 🌮.

    Sometimes you may want to add a new field or modify an existing one.
    

    Args:
        taco (Union[str, pathlib.Path]): The path to
            the TACO file.
        dataframe (tacotoolbox.datamodel.Collection): The new 
            dataframe of the TACO file.
            
    Returns:
        pathlib.Path: Path to the updated TACO file.
    """

    # Check if the taco file exists
    taco = pathlib.Path(taco)
    if not taco.exists():
        raise FileNotFoundError(f"The TACO file 🌮 '{taco}' does not exist.")
    
    # Drop the internal:* fields    
    dataframe.drop(
        columns=[
            col for col in dataframe.columns if (col.startswith("internal:") or col == "geometry")
        ],
        inplace=True
    )

    # Get the position of the collection
    # Create an in-memory Parquet file with BufferOutputStream
    with pa.BufferOutputStream() as sink:
        pq.write_table(
            pa.Table.from_pandas(dataframe),
            sink,
            compression="zstd",  # Highly efficient codec
            compression_level=22,  # Maximum compression for Zstandard
            use_dictionary=False,  # Optimizes for repeated values
        )
        # return a blob of the in-memory Parquet file as bytes
        # This is the FOOTER metadata
        FOOTER: bytes = sink.getvalue().to_pybytes()
    
    # Define the new FOOTER length
    newFL: bytes = len(FOOTER).to_bytes(8, "little")    

    if tortilla_or_taco(taco) == "🫓":        
        with open(taco, "r+b") as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
                # Modify the FL
                FO: int = int.from_bytes(mm[2:10], "little")
                mm[10:18] = newFL

                # Truncate the file if the new metadata is smaller or larger
                mm.resize(FO + len(FOOTER))

                # Overwrite the FOOTER
                mm[FO:] = FOOTER
        print(f"Footer updated successfully! 🫓")

    elif tortilla_or_taco(taco) == "🌮":
        with open(taco, "r+b") as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
                
                # Load FOOTER and COLLECTION offsets
                FO: int = int.from_bytes(mm[2:10], "little")
                CO = int.from_bytes(mm[26:34], byteorder="little")
                CL = int.from_bytes(mm[34:42], byteorder="little")

                # Load the COLLECTION bytes
                COLLECTION: bytes = mm[CO:CO+CL]
                
                # Upgrade the FOOTER length
                mm[10:18] = newFL

                # Upgrade the COLLECTION offset
                mm[26:34] = (FO + len(FOOTER)).to_bytes(8, byteorder="little")

                # Truncate the file if the new data is smaller or larger
                mm.resize(FO + len(FOOTER) + len(COLLECTION))

                # Write the FOOTER
                mm[FO:(FO+len(FOOTER))] = FOOTER

                # Write the COLLECTION
                mm[(FO+len(FOOTER)):] = COLLECTION
        
        print(f"Footer updated successfully! 🌮")
    else:
        raise ValueError("Invalid file type: must be a TACO 🌮 or a Tortilla 🫓")
    
    return taco


def tortilla_or_taco(taco: Union[str, pathlib.Path]) -> str:
    """ This function checks if a file is a Tortilla or a TACO.

    Args:
        taco (Union[str, pathlib.Path]): The path to the file.

    Returns:
        str: The type of file.
    """
    with open(taco, "r+b") as file:
        magic = file.read(2)
        if magic == b"#y":
            return "🫓"
        elif magic == b"WX":
            return "🌮"
        else:
            return "unknown"
