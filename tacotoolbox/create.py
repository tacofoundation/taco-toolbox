import os
import pathlib
import pytortilla

from tacotoolbox.datamodel import TACOCollection
from typing import Union, List


def create(
    metadata: TACOCollection,
    samples: pytortilla.datamodel.Samples,
    output: Union[str, pathlib.Path],
    nworkers: int = min(4, os.cpu_count()),
    chunk_size: str = "20GB",
    chunk_size_iter: str = "100MB",
    quiet: bool = False,
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Create a TACO file 🌮

    A TACO is a new simple format for storing same format files
    optimized for very fast random access.

    Args:
        samples (Samples): The list of samples to be included in
            the TACO file. All samples must have the same format
            (same extension). The Sample objects must have a unique
            `id` field.
        output (Union[str, pathlib.Path]): The path where the TACO
            file will be saved.
        nworkers (int, optional): The number of workers to use when writing
            the TACO. Defaults to 4.
        chunk_size (str, optional): Avoid large TACO files by splitting
            the data into chunks. By default, if the number of samples exceeds
            20GB, the data will be split into chunks of 20GB.
        chunk_size_iter (int, optional): The writting chunk size. By default,
            it is 100MB. Faster computers can use a larger chunk size.
        quiet (bool, optional): If True, the function does not print any
            message. By default, it is False.

    Returns:
        Union[pathlib.Path, List[pathlib.Path]]: The path of the TACO file.
            If the TACO file is split into multiple parts, a list of paths
            is returned.
    """

    # 1. Create Tortilla files    
    tortillas = pytortilla.create(
        samples=samples,
        output=output,
        nworkers=nworkers,
        chunk_size=chunk_size,
        chunk_size_iter=chunk_size_iter,
        quiet=quiet
    )

    # 2. Convert the tortillas in TACOs
    if isinstance(tortillas, list):
        tacos = []
        for tortilla in tortillas:
            taco = tortilla2taco(tortilla, metadata)            
            tacos.append(taco)
    else:
        tacos = tortilla2taco(tortillas, metadata)
    
    return tacos


def tortilla2taco(
    tortilla: Union[str, pathlib.Path],
    metadata: TACOCollection    
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Convert a Tortilla file 🫓 to a TACO file 🌮.

    Args:
        tortilla (Union[str, pathlib.Path]): The path to 
            the Tortilla file.
        metadata (tacotoolbox.datamodel.TACOCollection): The metadata 
            of the TACO file.
    """
    # Check if the Tortilla file exists
    tortilla = pathlib.Path(tortilla)
    if not tortilla.exists():
        raise FileNotFoundError(f"The file {tortilla} does not exist.")

    # Get the length of the Tortilla file
    tortilla_size: int = tortilla.stat().st_size
    tortilla_size_b: bytes = tortilla_size.to_bytes(8, byteorder="little")        

    # Convert the TACOcollection to a dictionary
    metadata_bytes: bytes = metadata.model_dump_json().encode()    
    metadata_size: int = len(metadata_bytes)
    metadata_size_b: bytes = metadata_size.to_bytes(8, byteorder="little")

    # 1. Upgrade the offset and length
    with open(tortilla, "r+b") as file:
        # Skip the Tortilla header
        file.seek(50)
        
        # write the CO (Collection Offset)
        file.write(tortilla_size_b)

        # write the CL (Collection Length)
        file.write(metadata_size_b)

        # Seek to the end of the file
        file.seek(0, os.SEEK_END)

        # Write the metadata
        file.write(metadata_bytes)

    return tortilla
