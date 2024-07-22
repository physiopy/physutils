import pydra

from physutils.io import load_physio
from physutils.physio import Physio


@pydra.mark.task
def transform_to_physio(input_file: str) -> Physio:
    physio_obj = load_physio(input_file)
    return physio_obj
