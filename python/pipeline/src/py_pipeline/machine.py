"""Machine information for the run metadata."""

__all__ = ["MachineInfo", "read_machine_info"]

import winreg
from dataclasses import dataclass

_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
_CPU_VALUE = "ProcessorNameString"


@dataclass(frozen=True)
class MachineInfo:
    """Machine values written to `meta.json`."""

    cpu: str
    gpu: str | None


def _read_cpu_name() -> str:
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _CPU_KEY) as key:
        value, _ = winreg.QueryValueEx(key, _CPU_VALUE)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("ProcessorNameString is empty")
    return value.strip()


def read_machine_info() -> MachineInfo:
    """Read the names of the machine devices used by the run."""
    return MachineInfo(cpu=_read_cpu_name(), gpu=None)
