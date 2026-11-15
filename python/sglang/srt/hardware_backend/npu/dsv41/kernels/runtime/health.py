"""Read-only physical-device checks for standalone operator qualifications."""

import os
import re
import subprocess
from pathlib import Path


def health_snapshot(physical: int, output: Path) -> dict:
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(physical):
        raise ValueError(
            "bind exactly the selected physical NPU before starting Python"
        )
    if os.environ.get("ASCEND_DEVICE_ID") != "0":
        raise ValueError("map the physical NPU to logical ASCEND_DEVICE_ID=0")
    health = subprocess.check_output(["npu-smi", "info"], text=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "health-before.txt").write_text(health)
    devices, current = {}, None
    for line in health.splitlines():
        first = re.search(r"\|\s*(\d+)\s+(910B\w*)\s*\|\s*(\w+)\s*\|", line)
        if first:
            current = int(first[1])
            devices[current] = {"name": first[2], "health": first[3]}
        second = re.search(
            r"\|\s*\d+\s*\|\s*[\w:.]+\s*\|\s*(\d+)\s+"
            r"\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|",
            line,
        )
        if second and current is not None:
            devices[current].update(
                utilization=int(second[1]),
                used_mb=int(second[2]),
                total_mb=int(second[3]),
            )
    card = devices.get(physical, {})
    if card.get("health") != "OK" or card.get("utilization") != 0:
        raise RuntimeError(f"physical NPU {physical} must be healthy and idle: {card}")
    if f"No running processes found in NPU {physical}" not in health:
        raise RuntimeError(f"physical NPU {physical} has active or unknown processes")
    if card.get("total_mb", 0) - card.get("used_mb", 0) <= 64:
        raise RuntimeError("insufficient free NPU memory")
    return {"physical_device": physical, "logical_device": 0, "all_cards": devices}
