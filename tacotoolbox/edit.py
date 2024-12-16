from typing import Union, List
import pathlib
import mmap
import tacotoolbox.datamodel


def collection(
    taco: Union[str, pathlib.Path],
    collection: tacotoolbox.datamodel.Collection    
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Edit the Collection of a TACO file 🌮.

    Args:
        taco (Union[str, pathlib.Path]): The path to
            the TACO file.
        collection (tacotoolbox.datamodel.Collection): The new 
            collection of the TACO file.
            
    """
    # Check if the taco file exists
    taco = pathlib.Path(taco)
    if not taco.exists():
        raise FileNotFoundError(f"The file {taco} does not exist.")

    # Convert the TACOcollection to a dictionary
    metadata_bytes: bytes = collection.model_dump_json().encode()    
    metadata_size: int = len(metadata_bytes)
    metadata_size_b: bytes = metadata_size.to_bytes(8, byteorder="little")

    # Upgrade the offset and length    
    with open(taco, "r+b") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
            
            # Read the old collection length (CL)
            old_CL_bytes = mm[34:42]
            old_CL = int.from_bytes(old_CL_bytes, byteorder="little")

            # Update the collection length (CL)
            mm[34:42] = metadata_size_b

            # Move the file pointer to the start of the old metadata and overwrite it
            mm.seek(old_CL)
            mm.write(metadata_bytes)

            # Truncate the file if the new metadata is smaller than the old one
            if metadata_size < old_CL:
                mm.resize(mm.tell() + metadata_size)

    return taco