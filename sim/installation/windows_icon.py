"""Embed the Trainer icon in distlib's GUI launcher before its script is appended."""

from __future__ import annotations

import ctypes
import struct
import tempfile
from ctypes import wintypes
from pathlib import Path


def icon_resources(icon: bytes) -> list[tuple[int, int, bytes]]:
    """Convert ICO offsets to Win32 RT_ICON and RT_GROUP_ICON resource IDs."""
    if len(icon) < 6:
        raise ValueError("Truncated Trainer ICO header")
    reserved, kind, count = struct.unpack_from("<HHH", icon)
    directory_end = 6 + 16 * count
    if reserved != 0 or kind != 1 or not count or len(icon) < directory_end:
        raise ValueError("Invalid Trainer ICO directory")
    group = bytearray(icon[:6])
    resources = []
    for index in range(count):
        entry = icon[6 + 16 * index : 22 + 16 * index]
        size, offset = struct.unpack_from("<II", entry, 8)
        if not size or offset < directory_end or offset + size > len(icon):
            raise ValueError("Invalid Trainer ICO image bounds")
        identifier = index + 1
        group.extend(entry[:12] + struct.pack("<H", identifier))
        resources.append((3, identifier, icon[offset : offset + size]))
    # distlib's stock GUI launcher uses group 101, neutral language, image IDs 1–7.
    resources.append((14, 101, bytes(group)))
    return resources


def launcher_with_icon(launcher: bytes, icon_path: Path, work_dir: Path) -> bytes:
    """Use Windows' resource writer on a bare PE; preserve distlib's later ZIP payload."""
    resources = icon_resources(icon_path.read_bytes())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    begin = kernel.BeginUpdateResourceW
    begin.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    begin.restype = wintypes.HANDLE
    update = kernel.UpdateResourceW
    update.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, wintypes.WORD, ctypes.c_void_p, wintypes.DWORD]
    update.restype = wintypes.BOOL
    end = kernel.EndUpdateResourceW
    end.argtypes = [wintypes.HANDLE, wintypes.BOOL]
    end.restype = wintypes.BOOL
    with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
        executable = Path(temporary) / "trainer-launcher.exe"
        executable.write_bytes(launcher)
        handle = begin(str(executable), False)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        discard = True
        try:
            for kind, identifier, data in resources:
                buffer = ctypes.create_string_buffer(data)
                if not update(handle, kind, identifier, 0, buffer, len(data)):
                    raise ctypes.WinError(ctypes.get_last_error())
            discard = False
        finally:
            if not end(handle, discard) and not discard:
                raise ctypes.WinError(ctypes.get_last_error())
        return executable.read_bytes()
