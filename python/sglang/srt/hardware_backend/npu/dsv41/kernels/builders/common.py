"""Shared offline build I/O and bounded compiler process groups."""

import hashlib
import json
import os
import signal
import subprocess
import time

MAX_BUILD_SECONDS = 300


def run(command, log, deadline):
    """One compiler process group; preserve diagnostics on failure or timeout."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timeout = max(0.01, deadline - time.monotonic())
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
        log.write_text(output)
        if isinstance(error, subprocess.TimeoutExpired):
            raise TimeoutError(
                f"operator build exceeded {timeout:.0f}s: {log}"
            ) from None
        raise
    log.write_text(output)
    if process.returncode:
        raise RuntimeError(f"build failed; see {log}\n{output[-8000:]}")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
