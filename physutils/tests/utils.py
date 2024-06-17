"""
Utilities for testing
"""

from os.path import join as pjoin

import numpy as np
from pkg_resources import resource_filename


def get_test_data_path(fname=None):
    """Function for getting `peakdet` test data path"""
    path = resource_filename("physutils", "tests/data")
    return pjoin(path, fname) if fname is not None else path


def get_sample_data():
    """Function for generating tiny sine wave form for testing"""
    data = np.sin(np.linspace(0, 20, 40))
    peaks, troughs = np.array([3, 15, 28]), np.array([9, 21, 34])

    return data, peaks, troughs
