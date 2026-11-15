"""Prepare finite variants and explicitly requested geometry through DeepJIT."""

import json
from contextlib import contextmanager
from pathlib import Path
from threading import local

_batch_state = local()


@contextmanager
def prepare_batch(operators=None):
    """Compile independent prepare/bind entries together, then load on exit.

    Only use with preparation calls: execution cannot consume a handle until
    this scope finishes. Inner batches share the outer producer/worker budget.
    """
    if getattr(_batch_state, "active", False):
        yield
        return
    if operators is None:
        from .native import load_library

        operators = load_library()
    operators.begin_prepare_batch()
    _batch_state.active = True
    try:
        yield
    finally:
        _batch_state.active = False
        operators.end_prepare_batch()


def _configuration_calls(operators, components, configurations):
    """Validate component and argument coverage before preparing any entry."""
    if not isinstance(configurations, dict):
        raise ValueError(
            "precompile configuration must map components to lists of geometry objects"
        )
    unknown = configurations.keys() - set(components)
    if unknown:
        raise ValueError(
            f"precompile configuration contains unselected JIT components: {sorted(unknown)}"
        )
    calls = []
    for component in components:
        function = getattr(operators, f"precompile_{component}")
        arguments = function.default._schema.arguments
        accepted = {argument.name for argument in arguments}
        required = {
            argument.name for argument in arguments if not argument.has_default_value()
        }
        variants = configurations.get(component, [{}])
        if not isinstance(variants, list) or not variants:
            raise ValueError(
                f"{component}: provide a nonempty list of actual geometry objects"
            )
        for geometry in variants:
            if not isinstance(geometry, dict):
                raise ValueError(
                    f"{component}: each geometry must be an object of named arguments"
                )
            if geometry.keys() - accepted:
                raise ValueError(
                    f"{component}: unknown geometry arguments {sorted(geometry.keys() - accepted)}"
                )
            if required - geometry.keys():
                raise ValueError(
                    f"{component}: missing actual geometry {sorted(required - geometry.keys())}; "
                    "provide --precompile-config or prepare the operator with real inputs before capture"
                )
            calls.append((component, function, geometry))
    return calls


def precompile(directory, configurations=None):
    if isinstance(configurations, (str, Path)):
        configurations = json.loads(Path(configurations).read_text())
    if configurations is None:
        configurations = {}
    import torch
    import torch_npu  # noqa: F401

    from .native import load_library

    directory = Path(directory).resolve()
    publication = json.loads((directory / "manifest.json").read_text())
    inventory = json.loads(
        (Path(publication["kernel_source_root"]) / "native/components.json").read_text()
    )
    backends = {item["name"]: item["backend"] for item in inventory["components"]}
    torch.npu.set_device(0)
    # The build alias may advance after this manifest was read. Prepare the
    # exact publication whose component inventory/source snapshot we selected.
    operators = load_library(publication["library"])
    components = [name for name in publication["components"] if backends[name] == "jit"]
    calls = _configuration_calls(operators, components, configurations)
    with prepare_batch(operators):
        for _, function, geometry in calls:
            function(**geometry)
