"""Geometry requests use real Torch schemas and never expand shape products."""

import pytest
import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.precompile import (
    _configuration_calls,
)


@pytest.fixture(scope="module")
def operators():
    library = torch.library.Library("dsv41_precompile_contract", "DEF")
    library.define("precompile_fixed() -> ()")
    library.define("precompile_geometry(int width, int mode=0) -> ()")
    yield torch.ops.dsv41_precompile_contract
    # Keep schema registrations alive for the complete fixture lifetime.
    del library


def test_actual_geometries_preserve_requested_pairs(operators):
    geometries = [{"width": 640, "mode": 2}, {"width": 1280}]
    plan = _configuration_calls(
        operators, ["fixed", "geometry"], {"geometry": geometries}
    )
    assert [(name, geometry) for name, _, geometry in plan] == [
        ("fixed", {}),
        ("geometry", geometries[0]),
        ("geometry", geometries[1]),
    ]
    assert plan[0][1].default._schema.name.endswith("::precompile_fixed")
    assert plan[1][1].default._schema.name.endswith("::precompile_geometry")


def test_missing_geometry_does_not_fall_back_to_an_invented_shape(operators):
    with pytest.raises(ValueError, match="missing actual geometry.*width"):
        _configuration_calls(operators, ["fixed", "geometry"], {})


@pytest.mark.parametrize(
    "configuration",
    [
        [],
        {"unselected": [{}]},
        {"geometry": {}},
        {"geometry": []},
        {"geometry": [[]]},
        {"geometry": [{"widht": 640}]},
        {"geometry": [{"width": 640, "unexpected": 1}]},
    ],
)
def test_invalid_requests_fail_before_producing_a_prepare_plan(
    operators, configuration
):
    with pytest.raises(ValueError):
        _configuration_calls(operators, ["fixed", "geometry"], configuration)
