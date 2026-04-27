Thermal Plate Simulator v13
===========================

This package is intended to run on:

- Windows
- macOS
- Linux

Main files
----------

- run_thermal_sim.py              Cross-platform launcher
- thermal_plate_sim_v13_gui.py    GUI
- thermal_core.py                 Simulation engine
- requirements.txt                Python dependencies

Quick start
-----------

Windows PowerShell:

    py -3 run_thermal_sim.py

or double-click:

    run_windows.bat


macOS Terminal:

    python3 run_thermal_sim.py

or double-click:

    run_macos.command

If macOS blocks the .command file, right-click it and choose Open. You can also run:

    chmod +x run_macos.command
    ./run_macos.command


Linux Terminal:

    python3 run_thermal_sim.py

or:

    ./run_linux.sh


Dependencies
------------

The launcher checks for NumPy and Matplotlib and can offer to install them.

Manual install:

    python -m pip install -r requirements.txt

On macOS:

    python3 -m pip install -r requirements.txt

Python from python.org is usually the simplest option on macOS because it includes a compatible Tkinter/Tk GUI stack.

If you use Homebrew Python and Tkinter is missing, try:

    brew install python-tk

On Debian/Ubuntu/Linux Mint, if Tkinter is missing:

    sudo apt install python3-tk

On Fedora:

    sudo dnf install python3-tkinter

On Arch/EndeavourOS:

    sudo pacman -S tk


v13 changes
--------------------------

- macOS Tk warning suppression.
- Better initial window sizing for laptop screens.
- Mouse wheel support for:
  - Windows wheel
  - macOS trackpad/wheel
  - Linux/X11 Button-4/Button-5 scrolling
- Native-ish menu bar with File/Help.
- Keyboard shortcuts:
  - macOS: Command-S save, Command-O load, Command-R run, Esc cancel
  - Windows/Linux: Ctrl-S save, Ctrl-O load, Ctrl-R run, Esc cancel
- Cross-platform launcher that checks Python, Tkinter, NumPy, and Matplotlib.
- 3D heatmap viewer with base plate, fins, and resistor blocks.
- 3D resistor side display: front, back/fin side, or both sides.
- Fin layout designer for individual fin center positions and heights.
- Fin presets: even, edge-biased, and place fins near resistor heat sources.


Notes
-----

The simulator is still an engineering approximation. For real resistor dump hardware, use a temperature sensor, fuses, and thermal cutoff.


v13 changes
-----------
- Improved 3D resistor visibility:
  - transparent plate mode
  - exploded view mode
  - resistor outlines drawn last for easier visibility from awkward angles
- Improved fin heat-transfer resolution:
  - fins are split into local thermal segments along the run length
  - the solver applies cooling locally per segment instead of one coarse full-fin strip
  - the 3D viewer colors fin segments from local plate temperature
- Heatsink dialog now includes "Thermal segments":
  - auto = recommended
  - a number forces that many segments per fin


v13 changes
-----------
- The 3D view now uses the same heatmap colormap as the 2D view.
- The 3D view now also follows the same fixed-scale temperature range logic as the 2D view.
- This keeps 2D and 3D colors consistent for the same snapshot.
