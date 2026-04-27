#!/usr/bin/env python3
"""
Cross-platform launcher for Thermal Plate Simulator v10.

It checks dependencies and starts the GUI. Works on Windows, macOS, and Linux.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import traceback

APP_FILE = "thermal_plate_sim_v10_gui.py"
REQUIRED = ["numpy", "matplotlib"]


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def main() -> int:
    system = platform.system()

    if sys.platform == "darwin":
        os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is recommended.")
        print(f"Current Python: {sys.version}")
        return 1

    try:
        import tkinter  # noqa: F401
    except Exception:
        print("Tkinter is not available in this Python installation.")
        if system == "Darwin":
            print("On macOS, the easiest fix is to install Python from python.org, which includes Tk support.")
            print("With Homebrew Python, try: brew install python-tk")
        elif system == "Linux":
            print("On Debian/Ubuntu/Mint, try: sudo apt install python3-tk")
            print("On Fedora, try: sudo dnf install python3-tkinter")
            print("On Arch, try: sudo pacman -S tk")
        else:
            print("On Windows, reinstall Python and make sure Tcl/Tk is included.")
        return 1

    missing = [m for m in REQUIRED if not has_module(m)]
    if missing:
        print("Missing Python packages:", ", ".join(missing))
        cmd = [sys.executable, "-m", "pip", "install", *missing]
        print("Install command:")
        print(" ".join(cmd))
        answer = input("Install them now? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes", "j", "ja"):
            try:
                subprocess.check_call(cmd)
            except Exception:
                print("Automatic install failed. Run the install command manually.")
                return 1
        else:
            return 1

    try:
        import thermal_plate_sim_v10_gui
        thermal_plate_sim_v10_gui.main()
        return 0
    except Exception:
        print("The simulator crashed during startup:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
