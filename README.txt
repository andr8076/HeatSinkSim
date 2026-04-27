Thermal Plate Simulator v8

Run:
    python thermal_plate_sim_v8_gui.py

Install dependencies if needed:
    python -m pip install numpy matplotlib

Files:
    thermal_plate_sim_v8_gui.py
    thermal_core.py

v8 adds:
    - Optional heatsink / fins support via extra effective surface area
    - Multi-worker optimizer candidate heat-solves
    - Worker count field: 0 = automatic
    - Keeps the simple tabbed UI from v7
