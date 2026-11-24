"""The nodes' space metadata, from plain names and DOF types (no Isaac Sim needed).

unienv_isaacsim/space_metadata.py imports nothing from Isaac, but the package's __init__ imports the nodes (Isaac
Lab), so the module is loaded from its file. test_adaptor.py checks the nodes' spaces in Isaac Sim.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace
from unienv_interface.space.space_utils.metadata_utils import validate_metadata

_PATH = Path(__file__).resolve().parents[1] / "unienv_isaacsim" / "space_metadata.py"
_spec = importlib.util.spec_from_file_location("_unienv_isaacsim_space_metadata", _PATH)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def box(*shape: int, metadata: dict) -> BoxSpace:
    return BoxSpace(PyTorchComputeBackend, -np.inf, np.inf, torch.float32, shape=shape, metadata=metadata)


def test_module_does_not_import_isaac():
    tree = ast.parse(_PATH.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert {name.split(".")[0] for name in imported} <= {"__future__", "collections", "typing", "unienv_interface"}


@pytest.mark.parametrize(("dof_types", "position", "velocity"), [
    (["rotation", "translation"], ["rad", "m"], ["rad/s", "m/s"]),
    (None, None, None),  # unknown: no units
    (["rotation", None], None, None),  # one unknown: no units rather than a guess
])  # fmt: skip
def test_joint_state_metadata(dof_types, position, velocity):
    md = sm.joint_state_metadata(["shoulder", "slider"], dof_types)
    for key, units in (("joint_position", position), ("joint_velocity", velocity)):
        validate_metadata(box(4, 2, metadata=md[key]))
        assert md[key]["kind"] == "proprio" and md[key]["names"] == ["shoulder", "slider"]
        assert md[key].get("units") == units


def test_joint_position_target_metadata():
    md = sm.joint_position_target_metadata(["shoulder", "slider"], ["rotation", "translation"])
    validate_metadata(box(4, 2, metadata=md))
    assert md == {"kind": "action", "control": "joint_position", "mode": "absolute", "units": ["rad", "m"],
                  "names": ["shoulder", "slider"]}  # fmt: skip
    assert "units" not in sm.joint_position_target_metadata(["shoulder"])


def test_root_pose_metadata():
    md = sm.root_pose_metadata()
    validate_metadata(box(4, 7, metadata=md))
    assert md["kind"] == "pose" and md["frame"] == "world" and md["rotation"] == "quat_wxyz"
    assert md["names"] == ["x", "y", "z", "qw", "qx", "qy", "qz"] and md["units"] == ["m"] * 3 + ["normalized"] * 4
    assert "env origins" in md["label"]


def test_contact_force_metadata():
    md = sm.contact_force_metadata(["Cube", "Lid"], source="contact")
    validate_metadata(box(4, 2, 3, metadata=md))  # names: the last axis (x, y, z), bodies: the one before
    assert md == {"kind": "force", "units": "N", "frame": "world", "names": ["fx", "fy", "fz"],
                  "bodies": ["Cube", "Lid"], "source": "contact"}  # fmt: skip


def test_wrong_lengths_are_caught():
    with pytest.raises(ValueError):
        validate_metadata(box(4, 3, metadata=sm.joint_position_target_metadata(["a", "b"], ["rotation"] * 2)))
