Thermal Plate Simulator v15.2
===========================

This package is intended to run on:

- Windows
- macOS
- Linux

Main files
----------

- run_thermal_sim.py              Cross-platform launcher
- thermal_plate_sim_v15.2_1_gui.py    GUI
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


v15.2 changes
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


v15.2 changes
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


v15.2 changes
-----------
- The 3D view now uses the same heatmap colormap as the 2D view.
- The 3D view now also follows the same fixed-scale temperature range logic as the 2D view.
- This keeps 2D and 3D colors consistent for the same snapshot.


v15.2 changes
-----------
- Fin height temperature gradient in the 3D viewer:
  - each fin thermal segment is split into vertical slices
  - slices are colored using a straight-fin temperature equation
  - this shows base-to-tip cooling instead of one uniform fin color

- Separate resistor temperature estimates:
  - plate footprint temperature
  - estimated resistor case/body temperature
  - estimated internal resistor element temperature
  - editable Case→plate °C/W and Element→case °C/W fields

- Precision/effectiveness improvements:
  - reports hottest plate point coordinates
  - adds precision notes when grid resolution is too coarse for resistor footprints
  - 3D resistor blocks are colored by estimated case/body temperature


v15.2 changes
-----------
- Much faster fin cooling-map calculation:
  - fin thermal segments are applied by grid index slices instead of full boolean masks
  - automatic fin segmentation is less excessive by default
- 3D viewer performance controls:
  - Fast / Balanced / Detailed quality selector
  - Fast mode draws fewer fin segment boxes and fewer height slices
  - Detailed mode keeps the richer fin gradient
- 2D heatmap fin overlay:
  - optional "Show fins in 2D heatmap" checkbox
  - fins are shown as dashed geometry overlays without replacing heatmap colors


v15.2 changes
-------------
- Replaces reduced-detail fin rendering with batched 3D rendering.
- Full / batched mode keeps every fin segment and every height slice.
- Fins and resistors are drawn as large Poly3DCollection batches to remove most per-object overhead.
- Simplified mode remains only as an emergency fallback for very old/slow machines.


v15.2 changes
-------------
- Entire 3D rendering system replaced.
- The 3D viewer no longer uses Matplotlib mplot3d.
- New viewer uses a custom Tk Canvas projection renderer:
  - direct polygon rendering
  - fast yaw/elevation/zoom controls
  - mouse drag rotation
  - same heatmap colormap
  - fin geometry and height gradients still shown
- This is not a physically new solver; it is a faster engineering visualization layer.
