#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helper class for holding physiological data and associated metadata information."""

import logging
from functools import wraps

from loguru import logger

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


def task(func):
    """
    Fake task decorator to import when pydra is not installed/used.

    Parameters
    ----------
    func: function
        Function to run the wrapper around

    Returns
    -------
    function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
        LGR.debug(
            "Pydra is not installed, thus generate_physio is not available as a pydra task. Using the function directly"
        )

    return wrapper


def is_bids_directory(path_to_dir):
    """
    Check if a directory is a BIDS compliant directory.

    Parameters
    ----------
    path_to_dir : os.path or str
        Path to (supposed) BIDS directory

    Returns
    -------
    bool
        True if the given path is a BIDS directory, False is not.
    """
    try:
        from bids import BIDSLayout
    except ImportError:
        raise ImportError(
            "To use BIDS-based feature, pybids must be installed. Install manually or with `pip install physutils[bids]`"
        )
    try:
        # Attempt to create a BIDSLayout object
        _ = BIDSLayout(path_to_dir)
        return True
    except Exception as e:
        # Catch other exceptions that might indicate the directory isn't BIDS compliant
        logger.error(
            f"An error occurred while trying to load {path_to_dir} as a BIDS Layout object: {e}"
        )
        return False
