import logging

import pydra

from physutils.io import load_from_bids, load_physio
from physutils.physio import Physio

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


@pydra.mark.task
def transform_to_physio(input_file: str, mode="physio", fs=None) -> Physio:
    LGR.debug(f"Loading physio object from {input_file}")
    if not fs:
        fs = None

    if mode == "physio":
        if fs is not None:
            physio_obj = load_physio(input_file, fs=fs, allow_pickle=True)
        else:
            physio_obj = load_physio(input_file, allow_pickle=True)

    elif mode == "bids":
        physio_obj = load_from_bids(input_file)
    else:
        raise ValueError(f"Invalid transform_to_physio mode: {mode}")
    return physio_obj
