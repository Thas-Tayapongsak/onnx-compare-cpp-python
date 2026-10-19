"""Machine and process information for the run metadata."""

__all__ = ["MachineInfo", "read_machine_info"]

import ctypes
import winreg
from dataclasses import dataclass

_GIB = 1024**3
_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
_CPU_VALUE = "ProcessorNameString"


@dataclass(frozen=True)
class MachineInfo:
    """Machine and process values written to `meta.json`."""

    cpu: str
    gpu: str | None
    ram_gb: float
    vram_gb: float


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _read_cpu_name() -> str:
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _CPU_KEY) as key:
        value, _ = winreg.QueryValueEx(key, _CPU_VALUE)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("ProcessorNameString is empty")
    return value.strip()


def _read_ram_gb() -> float:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    process = ctypes.c_void_p(kernel32.GetCurrentProcess())
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    if not psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    ):
        error = ctypes.get_last_error()
        raise OSError(error, "GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize / _GIB


def read_machine_info() -> MachineInfo:
    """Read the machine and process values once for one run."""
    return MachineInfo(
        cpu=_read_cpu_name(),
        gpu=None,
        ram_gb=_read_ram_gb(),
        vram_gb=0.0,
    )
