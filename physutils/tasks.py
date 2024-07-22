import pydra

from physutils.io import load_history, load_physio
from physutils.physio import Physio


@pydra.mark.task
def transform_to_physio(input_file: str, mode="physio") -> Physio:
    if mode == "physio":
        physio_obj = load_physio(input_file)
    elif mode == "history":
        physio_obj = load_history(input_file)
    elif mode == "bids":
        # TODO: Implement BIDS loading once the bids-support branch is merged
        raise NotImplementedError("BIDS loading is not yet implemented")
    else:
        raise ValueError(f"Invalid transform_to_physio mode: {mode}")
    return physio_obj
