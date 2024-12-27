import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanity_check(
    sample_metadata: pd.DataFrame,
    read_function: Callable[[Union[str, bytes]], None],
    max_workers: int = 4,
    **kwargs,
) -> List[str]:
    """
    Perform a sanity check on a given taco file to validate its contents.

    Parameters:
        file (Union[str, pathlib.Path]): Path to the taco file.
        read_function (Callable[[Union[str, bytes]], None]): Function to read the contents of the file.
        max_workers (int): Number of threads for concurrent processing. Default is 4.
        **kwargs: Ignored keyword arguments.

    Returns:
        List[str]: A list of IDs that failed the sanity check.
    """
    # get kwargs arguments
    super_name = kwargs.get("super_name", "")

    # Load metadata
    if super_name == "":
        logging.info(f"The sanity check is starting for {len(sample_metadata)} items.")

    failed_ids = []
    tasks = []

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            executor.submit(
                process_entry, idx, sample_metadata, read_function, super_name
            )
            for idx in range(len(sample_metadata))
        ]
        for future in as_completed(tasks):
            result = future.result()
            if result:
                failed_ids.append(result)

    if failed_ids:
        logging.warning(f"Sanity check failed for {len(failed_ids)} items.")
    else:
        if super_name == "":
            logging.info("All items passed the sanity check.")

    return failed_ids


def process_entry(idx, sample_metadata, read_function, super_name):
    """Processes a single metadata entry."""
    result = sample_metadata.read(idx)
    sample_id = super_name + sample_metadata.iloc[idx]["tortilla:id"]
    try:
        if isinstance(result, str) or isinstance(result, bytes):
            read_function(result)
        elif isinstance(result, pd.DataFrame):
            return sanity_check(
                result, read_function, max_workers=1, super_name=sample_id
            )
        else:
            raise ValueError(f"Unsupported return type for entry {sample_id}.")
    except Exception:
        return sample_id
