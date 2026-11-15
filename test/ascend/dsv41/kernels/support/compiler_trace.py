"""Check process-level evidence that prepared native entries do not compile.

Python audit hooks cannot see DeepJIT's C++ posix_spawn calls. A strace execve
trace covers both paths. Compiler/linker version discovery is recorded separately
and is allowed; any invocation that can compile or link code fails qualification.
"""

import hashlib
import json
import re
from pathlib import Path


def check_compiler_trace(path):
    path = Path(path)
    invocations = []
    executions = 0
    for line in path.read_text().splitlines():
        match = re.search(r'execve\("([^"\\]+)", ', line)
        if match is None:
            continue
        executions += 1
        name = Path(match[1]).name
        if not (
            name
            in {
                "bisheng",
                "ccec",
                "cmake",
                "ninja",
                "cc",
                "c++",
                "ld",
                "ld.gold",
                "ld.lld",
                "lld",
            }
            or re.fullmatch(
                r"(?:[\w.-]+-)?(?:g\+\+|gcc|clang\+\+|clang)(?:-\d+)?", name
            )
        ):
            continue
        arguments = re.match(r"(\[.*?\]), ", line[match.end() :])
        if arguments is None:
            raise AssertionError(f"incomplete compiler invocation in trace: {line}")
        argv = json.loads(arguments[1])
        if argv[1:] != ["--version"]:
            raise AssertionError(
                f"prepared native path attempted compilation/linking: {argv}"
            )
        invocations.append(argv)
    if executions == 0:
        raise AssertionError("trace contains no execve evidence")
    return {
        "method": "strace -f execve; Python and C++ process creation",
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "version_probes": invocations,
        "traced_executions": executions,
        "compilation_or_linking_invocations": 0,
    }
