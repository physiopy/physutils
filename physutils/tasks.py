import logging
from functools import wraps

from bids import BIDSLayout
from loguru import logger

from physutils.io import load_from_bids, load_physio
from physutils.physio import Physio

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)

try:
    import pydra

    pydra_imported = True
except ImportError:
    pydra_imported = False


def mark_task(pydra_imported=pydra_imported):
    def decorator(func):
        if pydra_imported:
            # If the decorator exists, apply it
            @wraps(func)
            def wrapped_func(*args, **kwargs):
                logger.debug(f"Creating pydra task for {func.__name__}")
                return pydra.mark.task(func)(*args, **kwargs)

            return wrapped_func
        # Otherwise, return the original function
        return func

    return decorator


def is_bids_directory(directory):
    try:
        # Attempt to create a BIDSLayout object
        _ = BIDSLayout(directory)
        return True
    except Exception as e:
        # Catch other exceptions that might indicate the directory isn't BIDS compliant
        logger.error(
            f"An error occurred while trying to load {directory} as a BIDS Layout object: {e}"
        )
        return False


@mark_task(pydra_imported=pydra_imported)
def transform_to_physio(
    input_file: str, mode="physio", fs=None, bids_parameters=dict(), bids_channel=None
) -> Physio:
    if not pydra_imported:
        LGR.warning(
            "Pydra is not installed, thus transform_to_physio is not available as a pydra task. Using the function directly"
        )
    LGR.debug(f"Loading physio object from {input_file}")
    if not fs:
        fs = None

    if mode == "auto":
        if input_file.endswith((".phys", ".physio", ".1D", ".txt", ".tsv", ".csv")):
            mode = "physio"
        elif is_bids_directory(input_file):
            mode = "bids"
        else:
            raise ValueError(
                "Could not determine mode automatically, please specify mode"
            )
    if mode == "physio":
        if fs is not None:
            physio_obj = load_physio(input_file, fs=fs, allow_pickle=True)
        else:
            physio_obj = load_physio(input_file, allow_pickle=True)

    elif mode == "bids":
        if bids_parameters is {}:
            raise ValueError("BIDS parameters must be provided when loading from BIDS")
        else:
            physio_array = load_from_bids(input_file, **bids_parameters)
            physio_obj = physio_array[bids_channel]
    else:
        raise ValueError(f"Invalid transform_to_physio mode: {mode}")
    return physio_obj
