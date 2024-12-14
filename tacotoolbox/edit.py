from typing import Union, List
import pathlib
import mmap
import tacotoolbox.datamodel


def metadata(
    taco: Union[str, pathlib.Path],
    metadata: tacotoolbox.datamodel.TACOCollection    
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """Edit the metadata of a TACO file 🌮.

    Args:
        taco (Union[str, pathlib.Path]): The path to
            the TACO file.
        metadata (tacotoolbox.datamodel.TACOCollection): The new 
            metadata of the TACO file.
            
    """
    # Check if the taco file exists
    taco = pathlib.Path(taco)
    if not taco.exists():
        raise FileNotFoundError(f"The file {taco} does not exist.")

    # Convert the TACOcollection to a dictionary
    metadata_bytes: bytes = metadata.model_dump_json().encode()    
    metadata_size: int = len(metadata_bytes)
    metadata_size_b: bytes = metadata_size.to_bytes(8, byteorder="little")

    # Upgrade the offset and length    
    with open(taco, "r+b") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
            # Read and write new CL (Collection Length)
            old_CL_bytes = mm[58:66]
            mm[58:66] = metadata_size_b

            # From byte to integer
            old_CL = int.from_bytes(old_CL_bytes, byteorder="little")

            # Clean the old metadata and write the new one
            mm = mm[:old_CL]
            mm.write(metadata_bytes)
        
    return taco
