# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import pytest

from physutils import io, physio
from physutils.tests.utils import (
    create_random_bids_structure,
    filter_physio,
    get_test_data_path,
)


def test_load_physio(caplog):
    # try loading pickle file (from io.save_physio)
    pckl = io.load_physio(get_test_data_path("ECG.phys"), allow_pickle=True)
    assert isinstance(pckl, physio.Physio)
    assert pckl.data.size == 44611
    assert pckl.fs == 1000.0
    pckl = io.load_physio(get_test_data_path("ECG.phys"), fs=500.0, allow_pickle=True)
    assert caplog.text.count("WARNING") == 1
    assert pckl.fs == 500.0

    # try loading CSV file
    csv = io.load_physio(get_test_data_path("ECG.csv"))
    assert isinstance(csv, physio.Physio)
    assert np.allclose(csv, pckl)
    assert np.isnan(csv.fs)
    assert csv.history[0][0] == "physutils.io.load_physio"

    # try loading array
    arr = io.load_physio(np.loadtxt(get_test_data_path("ECG.csv")))
    assert caplog.text.count("WARNING") == 2
    assert isinstance(arr, physio.Physio)
    arr = io.load_physio(
        np.loadtxt(get_test_data_path("ECG.csv")),
        history=[("np.loadtxt", {"fname": "ECG.csv"})],
    )
    assert isinstance(arr, physio.Physio)

    # try loading physio object (and resetting dtype)
    out = io.load_physio(arr, dtype="float32")
    assert out.data.dtype == np.dtype("float32")
    assert out.history[0][0] == "np.loadtxt"
    assert out.history[-1][0] == "physutils.io.load_physio"
    with pytest.raises(TypeError):
        io.load_physio([1, 2, 3])


def test_load_from_bids():
    create_random_bids_structure("physutils/tests/data", recording_id="cardiac")
    phys_array = io.load_from_bids(
        "physutils/tests/data/bids-dir",
        subject="01",
        session="01",
        task="rest",
        run="01",
        recording="cardiac",
    )

    for col in phys_array.keys():
        assert isinstance(phys_array[col], physio.Physio)
        # The data saved are the ones after t_0 = -3s
        assert phys_array[col].data.size == 80000
        assert phys_array[col].fs == 10000.0
        assert phys_array[col].history[0][0] == "physutils.io.load_from_bids"


def test_load_from_bids_no_rec():
    create_random_bids_structure("physutils/tests/data")
    phys_array = io.load_from_bids(
        "physutils/tests/data/bids-dir",
        subject="01",
        session="01",
        task="rest",
        run="01",
    )

    for col in phys_array.keys():
        assert isinstance(phys_array[col], physio.Physio)
        # The data saved are the ones after t_0 = -3s
        assert phys_array[col].data.size == 80000
        assert phys_array[col].fs == 10000.0
        assert phys_array[col].history[0][0] == "physutils.io.load_from_bids"


def test_save_physio(tmpdir):
    pckl = io.load_physio(get_test_data_path("ECG.phys"), allow_pickle=True)
    out = io.save_physio(tmpdir.join("tmp").purebasename, pckl)
    assert os.path.exists(out)
    assert isinstance(io.load_physio(out, allow_pickle=True), physio.Physio)


def test_load_history(tmpdir):
    # get paths of data, new history
    fname = get_test_data_path("ECG.csv")
    temp_history = tmpdir.join("tmp").purebasename

    # make physio object and perform some operations
    phys = io.load_physio(fname, fs=1000.0)
    filt = filter_physio(phys, [5.0, 15.0], "bandpass")

    # save history to file and recreate new object from history
    path = io.save_history(temp_history, filt)
    replayed = io.load_history(path, verbose=True)

    # ensure objects are the same
    assert np.allclose(filt, replayed)
    assert filt.history == replayed.history
    assert filt.fs == replayed.fs


def test_save_history(tmpdir, caplog):
    # get paths of data, original history, new history
    fname = get_test_data_path("ECG.csv")
    orig_history = get_test_data_path("history.json")
    temp_history = tmpdir.join("tmp").purebasename

    # make physio object and perform some operations
    phys = physio.Physio(np.loadtxt(fname), fs=1000.0)
    io.save_history(temp_history, phys)
    assert caplog.text.count("WARNING") == 1  # no history = warning
    filt = filter_physio(phys, [5.0, 15.0], "bandpass")
    path = io.save_history(temp_history, filt)  # dump history=

    # load both original and new json and ensure equality
    with open(path, "r") as src:
        hist = json.load(src)
    with open(orig_history, "r") as src:
        orig = json.load(src)
    assert hist == orig
