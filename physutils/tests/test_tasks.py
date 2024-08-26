"""Tests for physutils.tasks and their integration."""

import os

import physutils.tasks as tasks
from physutils import physio


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
