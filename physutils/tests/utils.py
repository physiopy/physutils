"""
Utilities for testing
"""

from os.path import join as pjoin

import numpy as np
from pkg_resources import resource_filename
from scipy import signal

from physutils import physio


def get_test_data_path(fname=None):
    """Function for getting `peakdet` test data path"""
    path = resource_filename("physutils", "tests/data")
    return pjoin(path, fname) if fname is not None else path


def get_sample_data():
    """Function for generating tiny sine wave form for testing"""
    data = np.sin(np.linspace(0, 20, 40))
    peaks, troughs = np.array([3, 15, 28]), np.array([9, 21, 34])

    return data, peaks, troughs


@physio.make_operation()
def filter_physio(data, cutoffs, method, *, order=3):
    """
    Applies an `order`-order digital `method` Butterworth filter to `data`

    Parameters
    ----------
    data : Physio_like
        Input physiological data to be filtered
    cutoffs : int or list
        If `method` is 'lowpass' or 'highpass', an integer specifying the lower
        or upper bound of the filter (in Hz). If method is 'bandpass' or
        'bandstop', a list specifying the lower and upper bound of the filter
        (in Hz).
    method : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        The type of filter to apply to `data`
    order : int, optional
        Order of filter to be applied. Default: 3

    Returns
    -------
    filtered : :class:`peakdet.Physio`
        Filtered input `data`
    """

    _valid_methods = ["lowpass", "highpass", "bandpass", "bandstop"]

    data = physio.check_physio(data, ensure_fs=True)
    if method not in _valid_methods:
        raise ValueError(
            "Provided method {} is not permitted; must be in {}.".format(
                method, _valid_methods
            )
        )

    cutoffs = np.array(cutoffs)
    if method in ["lowpass", "highpass"] and cutoffs.size != 1:
        raise ValueError("Cutoffs must be length 1 when using {} filter".format(method))
    elif method in ["bandpass", "bandstop"] and cutoffs.size != 2:
        raise ValueError("Cutoffs must be length 2 when using {} filter".format(method))

    nyq_cutoff = cutoffs / (data.fs * 0.5)
    if np.any(nyq_cutoff > 1):
        raise ValueError(
            "Provided cutoffs {} are outside of the Nyquist "
            "frequency for input data with sampling rate {}.".format(cutoffs, data.fs)
        )

    b, a = signal.butter(int(order), nyq_cutoff, btype=method)
    filtered = physio.new_physio_like(data, signal.filtfilt(b, a, data))

    return filtered
