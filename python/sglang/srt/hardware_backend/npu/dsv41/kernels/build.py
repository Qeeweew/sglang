"""CMake host/AICPU builds and explicit DeepJIT preparation.

Importing this CLI never imports Torch, initializes NPU, or compiles a kernel.
"""

from __future__ import annotations

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--backend", choices=("native",), default="native")
    args, remaining = parser.parse_known_args(argv)
    from .builders.native import main as build

    build(remaining)


if __name__ == "__main__":
    main()
