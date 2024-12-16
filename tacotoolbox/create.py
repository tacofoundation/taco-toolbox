import os
import mmap
import pathlib
import pytortilla
from tacotoolbox.datamodel import Collection
from typing import Union, List
import random


def create(
    collection: Collection,
    samples: pytortilla.datamodel.Samples,
    output: Union[str, pathlib.Path],
    nworkers: int = min(4, os.cpu_count()),
    chunk_size: str = "20GB",
    chunk_size_iter: str = "100MB",
    quiet: bool = False,
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Create a TACO file 🌮

    A TACO is a simple format for storing large datasets that
    require partial reading and random access. The TACO format
    is based on the Tortilla format. See the TACO documentation
    for more information.

    Args:
        collection (tacotoolbox.datamodel.Collection): The global
            metadata of the TACO file. This is a pydantic data model
            that contains fields like `id`, `description`, `version`,
            etc. See the `tacotoolbox.datamodel.Collection` class for
            more information.
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
        tortilla_message=taco_message,
        quiet=quiet
    )

    # 2. Convert the tortillas in TACOs
    if isinstance(tortillas, list):
        tacos = []
        for tortilla in tortillas:
            taco = tortilla2taco(tortilla, collection)            
            tacos.append(taco)
    else:
        tacos = tortilla2taco(tortillas, collection)
    
    return tacos


def tortilla2taco(
    tortilla: Union[str, pathlib.Path],
    collection: Collection    
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Convert a Tortilla file 🫓 to a TACO file 🌮.

    Args:
        tortilla (Union[str, pathlib.Path]): The path to 
            the Tortilla file.
        collection (tacotoolbox.datamodel.Collection): The global metadata 
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
    metadata_bytes: bytes = collection.model_dump_json().encode()    
    metadata_size: int = len(metadata_bytes)
    metadata_size_b: bytes = metadata_size.to_bytes(8, byteorder="little")

    # 1. Upgrade the MB, and set the CO and CL
    with open(tortilla, "r+b") as file:
        # Upgrade the Magic Number (MB)
        # The day I first piloted my own EVA in Tokyo-3.
        file.write(b"WX")
                
        # Skip the Tortilla header
        file.seek(26)

        # write the CO (Collection Offset)
        file.write(tortilla_size_b)

        # write the CL (Collection Length)
        file.write(metadata_size_b)

        # Seek to the end of the file
        file.seek(0, os.SEEK_END)

        # Write the metadata
        file.write(metadata_bytes)

    return tortilla


def taco2tortilla(
    taco: Union[str, pathlib.Path]
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Convert a TACO file 🌮 to a Tortilla file 🫓.

    Args:
        taco (Union[str, pathlib.Path]): The path to 
            the TACO file.
    """
    # Check if the taco file exists
    taco = pathlib.Path(taco)
    if not taco.exists():
        raise FileNotFoundError(f"The file {taco} does not exist.")

    # Open the TACO file
    with open(taco, "r+b") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
            # Set the Magic Number (MB)
            # Don't forget 3.oct.11        
            mm[0:2] = b"#y"

            # Clean the CO
            CO_int = int.from_bytes(mm[26:34], byteorder="little") 
            mm[26:34] = b"\x00" * 8

            # Clean the CL
            mm[34:42] = b"\x00" * 8

        # Truncate the file to CO_int after closing mmap
        file.truncate(CO_int)

    return taco


def taco_message() -> str:
    """Get a random taco message"""

    taco_messages = [
        "Making a TACO",
        "Making a TACO 🌮",
        "Cooking a TACO",
        "Making a TACO 🌮",
        "Working on a TACO",
        "Working on a TACO 🌮",
        "Rolling out a TACO",
        "Rolling out a TACO 🌮",
        "Baking a TACO",
        "Baking a TACO 🌮",
        "Grilling a TACO",
        "Grilling a TACO 🌮",
        "Toasting a TACO",
        "Toasting a TACO 🌮",
    ]

    # Randomly accessing a message
    random_message = random.choice(taco_messages)
    return random_message
