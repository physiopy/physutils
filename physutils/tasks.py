#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helper class for holding physiological data and associated metadata information."""

import logging

from .io import load_from_bids, load_physio, load_misc
from .physio import Physio
from .utils import is_bids_directory

# from loguru import logger

try:
    from pydra.mark import task
except ImportError:
    from .utils import task


LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


@task
def generate_physio(
    input_file: str, mode="auto", fs=None, bids_parameters=dict(), col_physio_type=None
) -> Physio:
    """
    Load a physio object from either a BIDS directory or an exported physio object.

    Parameters
    ----------
    input_file : str
        Path to input file
    mode : 'auto', 'physio', or 'bids', optional
        Mode to operate with
    fs : None, optional
        Set or force set sapmling frequency (Hz).
    bids_parameters : dictionary, optional
        Dictionary containing BIDS parameters
    col_physio_type : int or None, optional
        Object to pick up in a BIDS array of physio objects.

    """
    LGR.info(f"Loading physio object from {input_file}")

    if mode == "auto":
        if input_file.endswith((".phys")):
            mode = "physio"
        elif is_bids_directory(input_file):
            mode = "bids"
        elif input_file.endswith((".txt", ".tsv", ".csv")):
            mode = "misc"
        else:
            raise ValueError(
                "Could not determine input mode automatically. Please specify it manually."
            )
    if mode == "physio":
        physio_obj = load_physio(input_file, allow_pickle=True)

    elif mode == "bids":
        if bids_parameters is {}:
            raise ValueError("BIDS parameters must be provided when loading from BIDS")
        else:
            physio_array = load_from_bids(input_file, **bids_parameters)
            physio_obj = (
                physio_array[col_physio_type] if col_physio_type else physio_array
            )
    elif mode == "misc":
        physio_obj = load_misc(input_file, fs=fs)
    else:
        raise ValueError(f"Invalid generate_physio mode: {mode}")

    return physio_obj
