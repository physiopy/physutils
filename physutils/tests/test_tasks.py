"""Tests for physutils.tasks and their integration."""

import os

import physutils.tasks as tasks
from physutils import physio
from physutils.tests.utils import create_random_bids_structure


def test_transform_to_physio_phys_file():
    """Test transform_to_physio task."""
    physio_file = os.path.abspath("physutils/tests/data/ECG.phys")
    task = tasks.transform_to_physio(input_file=physio_file, mode="physio")
    assert task.inputs.input_file == physio_file
    assert task.inputs.mode == "physio"
    assert task.inputs.fs is None

    task()

    physio_obj = task.result().output.out
    assert isinstance(physio_obj, physio.Physio)
    assert physio_obj.fs == 1000
    assert physio_obj.data.shape == (44611,)


def test_transform_to_physio_bids_file():
    """Test transform_to_physio task."""
    create_random_bids_structure("physutils/tests/data", recording_id="cardiac")
    bids_parameters = {
        "subject": "01",
        "session": "01",
        "task": "rest",
        "run": "01",
        "recording": "cardiac",
    }
    bids_dir = os.path.abspath("physutils/tests/data/bids-dir")
    task = tasks.transform_to_physio(
        input_file=bids_dir,
        mode="bids",
        bids_parameters=bids_parameters,
        bids_channel="cardiac",
    )

    assert task.inputs.input_file == bids_dir
    assert task.inputs.mode == "bids"
    assert task.inputs.fs is None
    assert task.inputs.bids_parameters == bids_parameters
    assert task.inputs.bids_channel == "cardiac"

    task()

    physio_obj = task.result().output.out
    assert isinstance(physio_obj, physio.Physio)