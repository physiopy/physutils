import logging

import pydra

from physutils.io import load_from_bids, load_physio
from physutils.physio import Physio

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


@pydra.mark.task
def transform_to_physio(
    input_file: str, mode="physio", fs=None, bids_parameters=dict(), bids_channel=None
) -> Physio:
    LGR.debug(f"Loading physio object from {input_file}")
    if not fs:
        fs = None

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
