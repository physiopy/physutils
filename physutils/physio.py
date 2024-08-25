# -*- coding: utf-8 -*-
"""
Helper class for holding physiological data and associated metadata information
"""

import inspect
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def make_operation(*, exclude=None):
    """
    Wrapper to make functions into Physio operations

    Wrapped functions should accept a :class:`peakdet.Physio` instance, `data`,
    as their first parameter, and should return a :class:`peakdet.Physio`
    instance

    Parameters
    ----------
    exclude : list, optional
        What function parameters to exclude from being stored in history.
        Default: 'data'
    """

    def get_call(func):
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            # exclude 'data', by default
            ignore = ["data"] if exclude is None else exclude

            # set name as the full module function name
            name = inspect.getmodule(func).__name__ + "." + func.__name__
            # name = func.__name__

            # grab parameters from `func` by binding signature
            sig = inspect.signature(func)
            params = sig.bind(data, *args, **kwargs).arguments

            # actually run function on data
            data = func(data, *args, **kwargs)

            # it shouldn't be, but don't bother appending to history if it is
            if data is None:
                return data

            # get parameters and sort by key name, dropping ignored items and
            # attempting to coerce any numpy arrays or pandas dataframes (?!)
            # into serializable objects; this isn't foolproof but gets 80% of
            # the way there
            provided = {k: params[k] for k in sorted(params.keys()) if k not in ignore}
            for k, v in provided.items():
                if hasattr(v, "tolist"):
                    provided[k] = v.tolist()

            # append everything to data instance history
            if isinstance(data, tuple):
                data[0]._history += [(name, provided)]
            else:
                data._history += [(name, provided)]

            return data

        return wrapper

    return get_call


def _get_call(*, exclude=None, serializable=True):
    """
    Returns calling function name and dict of provided arguments (name : value)

    Parameters
    ----------
    exclude : list, optional
        What arguments to exclude from provided argument : value dictionary.
        Default: ['data']
    serializable : bool, optional
        Whether to coerce argument values to JSON serializable form. Default:
        True

    Returns
    -------
    function: str
        Name of calling function
    provided : dict
        Dictionary of function arguments and provided values
    """

    exclude = ["data"] if exclude is None else exclude
    if not isinstance(exclude, list):
        exclude = [exclude]

    # get one function call up the stack (the bottom is _this_ function)
    calling = inspect.stack(0)[1]
    frame, function = calling.frame, calling.function

    # get all the args / kwargs from the calling function
    argspec = inspect.getfullargspec(frame.f_globals[function])
    args = sorted(argspec.args + argspec.kwonlyargs)

    # save arguments + argument values for everything not in `exclude`
    provided = {k: frame.f_locals[k] for k in args if k not in exclude}

    # if we want `provided` to be serializable, we can do a little cleaning up
    # this is NOT foolproof, but will coerce numpy arrays to lists which tends
    # to be the main issue with these sorts of things
    if serializable:
        for k, v in provided.items():
            if hasattr(v, "tolist"):
                provided[k] = v.tolist()

    function = inspect.getmodule(frame).__name__ + "." + function

    return function, provided


def check_physio(data, ensure_fs=True, copy=False):
    """
    Checks that `data` is in correct format (i.e., `peakdet.Physio`)

    Parameters
    ----------
    data : Physio_like
    ensure_fs : bool, optional
        Raise ValueError if `data` does not have a valid sampling rate
        attribute.
    copy: bool, optional
        Whether to return a copy of the provided data. Default: False

    Returns
    -------
    data : peakdet.Physio
        Loaded physio object

    Raises
    ------
    ValueError
        If `ensure_fs` is set and `data` doesn't have valid sampling rate
    """

    from physutils.io import load_physio

    if not isinstance(data, Physio):
        data = load_physio(data)
    if ensure_fs and np.isnan(data.fs):
        raise ValueError("Provided data does not have valid sampling rate.")
    if copy is True:
        return new_physio_like(
            data,
            data.data,
            copy_history=True,
            copy_metadata=True,
            copy_suppdata=True,
            copy_label=True,
            copy_physio_type=True,
            copy_computed_metrics=True,
        )
    return data


def new_physio_like(
    ref_physio,
    data,
    *,
    fs=None,
    suppdata=None,
    dtype=None,
    copy_history=True,
    copy_metadata=True,
    copy_suppdata=True,
    copy_label=True,
    copy_physio_type=True,
    copy_computed_metrics=True,
):
    """
    Makes `data` into physio object like `ref_data`

    Parameters
    ----------
    ref_physio : Physio_like
        Reference `Physio` object
    data : array_like
        Input physiological data
    fs : float, optional
        Sampling rate of `data`. If not supplied, assumed to be the same as
        in `ref_physio`
    suppdata : array_like, optional
        New supplementary data. If not supplied, assumed to be the same.
    dtype : data_type, optional
        Data type to convert `data` to, if conversion needed. Default: None
    copy_history : bool, optional
        Copy history from `ref_physio` to new physio object. Default: True
    copy_metadata : bool, optional
        Copy metadata from `ref_physio` to new physio object. Default: True
    copy_suppdata : bool, optional
        Copy suppdata from `ref_physio` to new physio object. Default: True
    copy_label : bool, optional
        Copy label from `ref_physio` to new physio object. Default: True
    copy_physio_type : bool, optional
        Copy physio_type from `ref_physio` to new physio object. Default: True
    copy_computed_metrics : bool, optional
        Copy computeed_metrics from `ref_physio` to new physio object. Default: True

    Returns
    -------
    data : peakdet.Physio
        Loaded physio object with provided `data`
    """

    if fs is None:
        fs = ref_physio.fs
    if dtype is None:
        dtype = ref_physio.data.dtype
    history = list(ref_physio.history) if copy_history else []
    metadata = dict(**ref_physio._metadata) if copy_metadata else None

    if suppdata is None:
        suppdata = ref_physio._suppdata if copy_suppdata else None

    label = ref_physio.label if copy_label else None
    physio_type = ref_physio.physio_type if copy_physio_type else None
    computed_metrics = (
        dict(ref_physio.computed_metrics) if copy_computed_metrics else {}
    )

    # make new class
    out = ref_physio.__class__(
        np.array(data, dtype=dtype),
        fs=fs,
        history=history,
        metadata=metadata,
        suppdata=suppdata,
        physio_type=physio_type,
        label=label,
    )
    out._computed_metrics = computed_metrics
    return out


class Physio:
    """
    Class to hold physiological data and relevant information

    Parameters
    ----------
    data : array_like
        Input data array
    fs : float, optional
        Sampling rate of `data` (Hz). Default: None
    history : list of tuples, optional
        Functions performed on `data`. Default: None
    metadata : dict, optional
        Metadata associated with `data`. Default: None
    suppdata : array_like, optional
        Support data array. Default: None

    Attributes
    ----------
    data : :obj:`numpy.ndarray`
        Physiological waveform
    fs : float
        Sampling rate of `data` in Hz
    history : list of tuples
        History of functions that have been performed on `data`, with relevant
        parameters provided to functions.
    peaks : :obj:`numpy.ndarray`
        Indices of peaks in `data`
    troughs : :obj:`numpy.ndarray`
        Indices of troughs in `data`
    suppdata : :obj:`numpy.ndarray`
        Secondary physiological waveform
    physio_type : {'respiratory', 'cardiac', None}
        Type of the contained physiological signal. Default: None
    label : string
        Label of the physiological signal
    """

    def __init__(
        self,
        data,
        fs=None,
        history=None,
        metadata=None,
        suppdata=None,
        physio_type=None,
        label=None,
    ):
        _supported_physio_types = ["respiratory", "cardiac", None]
        logger.debug("Initializing new Physio object")
        self._data = np.asarray(data).squeeze()
        if self.data.ndim > 1:
            raise ValueError(
                "Provided data dimensionality {} > 1.".format(self.data.ndim)
            )

        if not np.issubdtype(self.data.dtype, np.number):
            raise ValueError(
                "Provided data of type {} is not numeric.".format(self.data.dtype)
            )
        self._fs = np.float64(fs)
        self._physio_type = None if physio_type is None else physio_type
        if self.physio_type not in _supported_physio_types:
            raise ValueError(
                "Provided physiological signal type {} is not supported. It must be in {}".format(
                    self.physio_type, _supported_physio_types
                )
            )

        self._label = label
        self._history = [] if history is None else history
        if not isinstance(self._history, list) or any(
            [not isinstance(f, tuple) for f in self._history]
        ):
            raise TypeError(
                "Provided history {} must be a list-of-tuples. "
                "Please check inputs.".format(history)
            )
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError(
                    "Provided metadata {} must be dict-like.".format(metadata)
                )
            for k in ["peaks", "troughs", "reject"]:
                metadata.setdefault(k, np.empty(0, dtype=int))
                if not isinstance(metadata.get(k), np.ndarray):
                    try:
                        metadata[k] = np.asarray(metadata.get(k), dtype=int)
                    except TypeError:
                        raise TypeError(
                            "Provided metadata must be dict-like"
                            "with integer array entries."
                        )
            self._metadata = dict(**metadata)
        else:
            self._metadata = dict(
                peaks=np.empty(0, dtype=int),
                troughs=np.empty(0, dtype=int),
                reject=np.empty(0, dtype=int),
            )
        self._suppdata = None if suppdata is None else np.asarray(suppdata).squeeze()
        self._computed_metrics = dict()

    def __array__(self):
        return self.data

    def __getitem__(self, slicer):
        return self.data[slicer]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "{name}(size={size}, fs={fs})".format(
            name=self.__class__.__name__, size=self.data.size, fs=self.fs
        )

    __repr__ = __str__

    @property
    def data(self):
        """Physiological data"""
        return self._data

    @property
    def fs(self):
        """Sampling rate of data (Hz)"""
        return self._fs

    @property
    def history(self):
        """Functions that have been performed on / modified `data`."""
        return self._history

    @property
    def peaks(self):
        """Indices of detected peaks in `data`"""
        return self._masked.compressed()

    @property
    def troughs(self):
        """Indices of detected troughs in `data`"""
        return self._metadata["troughs"]

    @property
    def _masked(self):
        return np.ma.masked_array(
            self._metadata["peaks"],
            mask=np.isin(self._metadata["peaks"], self._metadata["reject"]),
        )

    @property
    def suppdata(self):
        """Physiological data"""
        return self._suppdata

    @property
    def label(self):
        """Physio instance label"""
        return self._label

    @property
    def physio_type(self):
        """Physiological signal type"""
        return self._physio_type

    @property
    def computed_metrics(self):
        """Physio object computed metrics (phys2denoise)"""
        return self._computed_metrics

    def plot_physio(self, *, ax=None):
        """
        Plots `Physio.data` and associated peaks / troughs

        Parameters
        ----------
        data : Physio_like
            Physiological data to plot
        ax : :class:`matplotlib.axes.Axes`, optional
            Axis on which to plot `data`. If None, a new axis is created. Default:
            None

        Returns
        -------
        ax : :class:`matplotlib.axes.Axes`
            Axis with plotted `Physio.data`
        """
        logger.debug(f"Plotting {self.label}")
        # generate x-axis time series
        fs = 1 if np.isnan(self.fs) else self.fs
        time = np.arange(0, len(self) / fs, 1 / fs)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # plot data with peaks + troughs, as appropriate
        ax.plot(
            time,
            self.data,
            "b",
            time[self.peaks],
            self[self.peaks],
            ".r",
            time[self.troughs],
            self[self.troughs],
            ".g",
        )

        return ax

    def phys2neurokit(
        self, copy_data, copy_peaks, copy_troughs, module, neurokit_path=None
    ):
        """Physio to neurokit dataframe

        Parameters
        ----------
        copy_data: bool
            whether to copy raw data from Physio object to dataframe
        copy_peaks: bool
            whether to copy peaks from Physio object to dataframe
        copy_troughs: bool
            whether to copy troughs from Physio object to dataframe
        module: string
            name of module (eg. 'EDA', 'RSP', 'PPG'...)
        neurokit_path: string
            path to neurokit dataframe
        """
        import pandas as pd

        if neurokit_path is not None:
            df = pd.read_csv(neurokit_path, sep="\t")
        else:
            df = pd.DataFrame(
                0,
                index=np.arange(len(self.data)),
                columns=["%s_Raw" % module, "%s_Peaks" % module, "%s_Troughs" % module],
            )

        if copy_data:
            df.loc[:, df.columns.str.endswith("Raw")] = self.data

        if copy_peaks:
            b_peaks = np.zeros(len(self.data))
            b_peaks[self.peaks] = 1
            df.loc[:, df.columns.str.endswith("Peaks")] = b_peaks

        if copy_troughs:
            b_troughs = np.zeros(len(self.data))
            b_troughs[self.troughs] = 1
            df.loc[:, df.columns.str.endswith("Troughs")] = b_troughs

        return df

    @classmethod
    def neurokit2phys(
        cls, neurokit_path, fs, copy_data, copy_peaks, copy_troughs, **kwargs
    ):
        """Neurokit dataframe to phys

        Parameters
        ----------
        neurokit_path: string
            path to neurokit dataframe
        fs: float
            sampling rate
        copy_data: bool
            whether to copy raw data from Physio object to dataframe
        copy_peaks: bool
            whether to copy peaks from Physio object to dataframe
        copy_troughs: bool
            whether to copy troughs from Physio object to dataframe
        suppdata: array_like, optional
            Support data array. Default: None
        """
        import pandas as pd

        df = pd.read_csv(neurokit_path, sep="\t")

        if copy_data:
            # if cleaned data exists, substitute 'data' with cleaned data, else use raw data
            if df.columns.str.endswith("Clean").any():
                data = np.hstack(df.loc[:, df.columns.str.endswith("Clean")].to_numpy())
            elif df.columns.str.endswith("Raw").any():
                data = np.hstack(df.loc[:, df.columns.str.endswith("Raw")].to_numpy())

        if copy_peaks:
            # if peaks exists
            if df.columns.str.endswith("Peaks").any():
                peaks = np.where(df.loc[:, df.columns.str.endswith("Peaks")] == 1)[0]

        if copy_troughs:
            # if troughs exists
            if df.columns.str.endswith("Troughs").any():
                troughs = np.where(df.loc[:, df.columns.str.endswith("Troughs")] == 1)[
                    0
                ]

        if "peaks" in locals() and "troughs" in locals():
            metadata = dict(peaks=peaks, troughs=troughs)
        elif "peaks" in locals() and "troughs" not in locals():
            metadata = dict(peaks=peaks)

        return cls(data, fs=fs, metadata=metadata, **kwargs)


class MRIConfig:
    """
    Class to hold MRI configuration information

    Parameters
    ----------
    slice_timings : 1D array_like
        Slice timings in seconds
    n_scans : int
        Number of volumes in the MRI scan
    tr : float
        Repetition time in seconds
    """

    def __init__(self, slice_timings=None, n_scans=None, tr=None):
        if np.ndim(slice_timings) > 1:
            raise ValueError("Slice timings must be a 1-dimensional array.")

        self._slice_timings = np.asarray(slice_timings)
        self._n_scans = int(n_scans)
        self._tr = float(tr)
        logger.debug(f"Initializing new MRIConfig object: {self}")

    def __str__(self):
        return "{name}(n_scans={n_scans}, tr={tr})".format(
            name=self.__class__.__name__,
            n_scans=self._n_scans,
            tr=self._tr,
        )

    @property
    def slice_timings(self):
        """Slice timings in seconds"""
        return self._slice_timings

    @property
    def n_scans(self):
        """Number of volumes in the MRI scan"""
        return self._n_scans

    @property
    def tr(self):
        """Repetition time in seconds"""
        return self._tr
