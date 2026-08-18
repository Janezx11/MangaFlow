"""Windows 原生文件夹选择对话框（ctypes 调 SHBrowseForFolderW）

用于 Gradio GUI 中点击按钮弹出系统文件夹选择器，替代网页内置的文件浏览面板。
仅 Windows 生效；其他平台回退返回空字符串（调用方应自行处理）。
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes

_lock = threading.Lock()

# SHBrowseForFolder 标志
_BIF_RETURNONLYFSDIRS = 0x00000001
_BIF_NEWDIALOGSTYLE = 0x00000040

# 回调消息 / 设置选择
_BFFM_INITIALIZED = 1
_BFFM_SETSELECTION = 0x0400 + 103  # WM_USER + 103


class _BROWSEINFO(ctypes.Structure):
    _fields_ = [
        ("hwndOwner", wintypes.HWND),
        ("pidlRoot", ctypes.c_void_p),
        ("pszDisplayName", wintypes.LPWSTR),
        ("lpszTitle", wintypes.LPCWSTR),
        ("ulFlags", wintypes.UINT),
        ("lpfn", ctypes.c_void_p),
        ("lParam", wintypes.LPARAM),
        ("iImage", ctypes.c_int),
    ]


_BFFCALLBACK = ctypes.WINFUNCTYPE(
    ctypes.c_int, wintypes.HWND, wintypes.UINT, wintypes.LPARAM, wintypes.LPARAM
)


def _make_callback(initial_dir: str):
    """构造设置初始目录的回调（BFFM_INITIALIZED 时发送 BFFM_SETSELECTION）"""
    user32 = ctypes.windll.user32
    user32.SendMessageW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
    ]
    user32.SendMessageW.restype = wintypes.LPARAM

    @_BFFCALLBACK
    def _cb(hwnd, u_msg, _l_param, _lp_data):
        if u_msg == _BFFM_INITIALIZED and initial_dir:
            # 传整型地址作为 lParam（指针不能 cast 成整型类型，先转 c_void_p 取地址值）
            lparam = ctypes.cast(ctypes.c_wchar_p(initial_dir), ctypes.c_void_p).value
            user32.SendMessageW(hwnd, _BFFM_SETSELECTION, 1, lparam)
        return 0

    return _cb


def pick_directory(initial_dir: str = "", title: str = "选择文件夹") -> str:
    """弹出 Windows 原生文件夹选择对话框。

    返回选中文件夹的绝对路径；用户取消则返回空字符串。
    非 Windows 平台返回空字符串。
    """
    if os.name != "nt":
        return ""

    # 串行化：避免多个线程同时弹对话框导致冲突
    with _lock:
        com_ok = False
        try:
            ctypes.windll.ole32.CoInitialize(None)
            com_ok = True
        except Exception:
            pass

        try:
            shell32 = ctypes.windll.shell32
            shell32.SHBrowseForFolderW.restype = ctypes.c_void_p  # PIDL 指针
            shell32.SHBrowseForFolderW.argtypes = [ctypes.POINTER(_BROWSEINFO)]
            shell32.SHGetPathFromIDListW.restype = wintypes.BOOL
            shell32.SHGetPathFromIDListW.argtypes = [
                ctypes.c_void_p, wintypes.LPWSTR,
            ]
            ctypes.windll.ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]

            display = ctypes.create_unicode_buffer(512)
            callback = _make_callback(initial_dir) if initial_dir else None

            bi = _BROWSEINFO(
                hwndOwner=None,
                pidlRoot=None,
                pszDisplayName=ctypes.cast(display, wintypes.LPWSTR),
                lpszTitle=title,
                ulFlags=_BIF_RETURNONLYFSDIRS | _BIF_NEWDIALOGSTYLE,
                lpfn=ctypes.cast(callback, ctypes.c_void_p) if callback else None,
                lParam=0,
                iImage=0,
            )
            pidl = shell32.SHBrowseForFolderW(ctypes.byref(bi))
            if not pidl:
                return ""
            try:
                path_buf = ctypes.create_unicode_buffer(1024)
                if shell32.SHGetPathFromIDListW(pidl, path_buf):
                    return path_buf.value
                return ""
            finally:
                ctypes.windll.ole32.CoTaskMemFree(pidl)
        finally:
            if com_ok:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass
