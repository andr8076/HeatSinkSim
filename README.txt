Thermal Plate Simulator v9

Run:
    python thermal_plate_sim_v9_gui.py

Install dependencies if needed:
    python -m pip install numpy matplotlib

Files:
    thermal_plate_sim_v9_gui.py
    thermal_core.py

v9 adds a geometry-based heatsink builder:
- fins are actual strips on the back side of the plate
- resistors are assumed to mount on the flat/front side
- fin count, orientation, thickness, run length, positions, and individual heights can be entered
- default placement is even
- fin thickness can be 'same' to use the base plate thickness
- run length can be 'full'
- positions can be 'even' or comma-separated center positions in cm
- heights can be 'same' or comma-separated mm values
- fin surface area and fin efficiency are calculated from dimensions
- fin cooling is applied locally under each fin footprint, so placement affects the heatmap

Important:
The dimensions and fin efficiency are calculated, but passive airflow h is still an environmental input. For real safety, test with a temperature sensor and use a cutoff.
