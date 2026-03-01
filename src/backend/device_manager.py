"""
Compatibility shim for legacy imports.

Several backend modules import `device_manager`, while the concrete
implementation currently lives in `device_utils.py`.
"""

from .device_utils import DeviceConfig, get_device_config, torch_device

__all__ = ["DeviceConfig", "get_device_config", "torch_device"]
