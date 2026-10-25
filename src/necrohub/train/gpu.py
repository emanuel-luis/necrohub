"""CUDA build check: does this torch wheel ship kernels for the installed GPU?"""

from typing import Any


def gpu_report() -> dict[str, Any]:
    """Torch version, CUDA build, device name, compute capability, the wheel's arch list and
    whether the device's sm_XY is in it. Device fields are None without a CUDA device."""
    import torch

    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": None,
        "capability": None,
        "arch_list": [],
        "arch_supported": False,
    }
    if report["cuda_available"]:
        major, minor = torch.cuda.get_device_capability(0)
        report["device"] = torch.cuda.get_device_name(0)
        report["capability"] = f"sm_{major}{minor}"
        report["arch_list"] = list(torch.cuda.get_arch_list())
        report["arch_supported"] = report["capability"] in report["arch_list"]
    return report


def gpu_problem(report: dict[str, Any]) -> str | None:
    """Why CUDA training cannot run with this torch build, or None when it can."""
    if not report["cuda_available"]:
        return (
            f"CUDA is not available (torch {report['torch']}, CUDA build {report['cuda_build']}): "
            "install the cu126 extra on a machine with an NVIDIA driver, or train on the CPU"
        )
    if not report["arch_supported"]:
        return (
            f"this torch build has no kernels for {report['capability']} ({report['device']}); "
            f"its arch list is {', '.join(report['arch_list'])}"
        )
    return None
