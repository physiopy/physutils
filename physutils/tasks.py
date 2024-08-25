import pydra

from physutils.io import load_from_bids, load_history, load_physio
from physutils.physio import Physio


@pydra.mark.task
def transform_to_physio(input_file: str, mode="physio") -> Physio:
    if mode == "physio":
        physio_obj = load_physio(input_file)
    elif mode == "history":
        physio_obj = load_history(input_file)
    elif mode == "bids":
        physio_obj = load_from_bids(input_file)
    else:
        raise ValueError(f"Invalid transform_to_physio mode: {mode}")
    return physio_obj
