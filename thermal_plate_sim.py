#!/usr/bin/env python3
"""
thermal_plate_sim_v4_gui.py

Desktop GUI for simulating passive heat spreading in a flat metal plate.

Adds an advanced cooling estimator for passive builds.

Features:
  - Tkinter UI
  - Multiple rectangular resistors / heat sources
  - Time-based transient simulation
  - Slider through time snapshots
  - Optional steady-state final heatmap
  - Advanced cooling estimator for orientation, enclosure, wall clearance, surface, air movement, hot-air path, and blockage
  - Config save/load as JSON
  - Export current heatmap PNG
  - Export current temperature grid CSV

Model:
  Thin metal plate, 2D heat spreading.
  Heat leaves both large faces by convection.
  Resistors inject heat over their rectangular contact footprints.

Important:
  This is an engineering approximation, not a certified thermal design tool.
  Use a real temperature sensor and thermal cutoff in the physical setup.
  The temperature shown is plate/contact temperature, not guaranteed internal resistor temperature.

Dependencies:
  Python 3
  numpy
  matplotlib

Install:
  python3 -m pip install numpy matplotlib

Run:
  python3 thermal_plate_sim_v4_gui.py
"""

from __future__ import annotations

import csv
import json
import math
import queue
import re
import threading
import time
import traceback
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


MATERIALS = {
    "aluminium": {"k": 205.0, "rho": 2700.0, "cp": 900.0},
    "copper":    {"k": 385.0, "rho": 8960.0, "cp": 385.0},
    "steel":     {"k": 45.0,  "rho": 7850.0, "cp": 470.0},
    "iron":      {"k": 80.0,  "rho": 7870.0, "cp": 450.0},
    "custom":    {"k": 205.0, "rho": 2700.0, "cp": 900.0},
}


@dataclass
class Resistor:
    name: str
    power_w: float
    center_x_cm: float
    center_y_cm: float
    length_mm: float
    width_mm: float


@dataclass
class PlateConfig:
    plate_length_cm: float
    plate_width_cm: float
    plate_thickness_mm: float
    material_name: str
    thermal_conductivity_w_mk: float
    density_kg_m3: float
    heat_capacity_j_kgk: float
    ambient_c: float
    convection_h_w_m2k: float
    grid_mm: float
    resistors: List[Resistor]
    initial_plate_temp_c: float
    max_time_s: float
    snapshot_every_s: float
    include_steady_state: bool

    # Advanced cooling estimate inputs. If disabled, convection_h_w_m2k is used directly.
    advanced_cooling_enabled: bool = False
    orientation: str = "vertical"
    environment: str = "open_air"
    wall_clearance_cm: float = 20.0
    surface_finish: str = "bare_metal"
    air_movement: str = "still_air"
    hot_air_path: str = "free_rise"
    blockage_percent: float = 0.0
    cooling_notes: str = ""


@dataclass
class SimulationSnapshot:
    label: str
    time_s: Optional[float]
    temp_c: np.ndarray
    is_steady_state: bool = False


@dataclass
class SimulationResult:
    cfg: PlateConfig
    x_m: np.ndarray
    y_m: np.ndarray
    dx_m: float
    dy_m: float
    q_w_m2: np.ndarray
    resistor_masks: List[Tuple[Resistor, np.ndarray, float]]
    snapshots: List[SimulationSnapshot]
    steady_state_temp_c: Optional[np.ndarray]
    summary: Dict


def parse_float(text: str, name: str, min_value: Optional[float] = None) -> float:
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{name} must be a number.")

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value}.")

    return value


def parse_time_to_seconds(text: str, name: str) -> float:
    """
    Accepts:
      30s, 1m, 1.5m, 2h
      or plain number = minutes
    """
    raw = str(text).strip().replace(",", ".")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(\s*[smhSMH]?)", raw)
    if not m:
        raise ValueError(f"{name} must be like 30s, 1m, 15m, or 1h.")

    value = float(m.group(1))
    unit = m.group(2).strip().lower()

    if unit == "s":
        seconds = value
    elif unit == "h":
        seconds = value * 3600.0
    else:
        seconds = value * 60.0

    if seconds <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return seconds


def format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "steady-state"

    if seconds < 60:
        return f"{seconds:.0f} s"

    if seconds < 3600:
        minutes = seconds / 60.0
        if abs(minutes - round(minutes)) < 1e-9:
            return f"{minutes:.0f} min"
        return f"{minutes:.1f} min"

    hours = seconds / 3600.0
    if abs(hours - round(hours)) < 1e-9:
        return f"{hours:.0f} h"
    return f"{hours:.2f} h"


def safe_time_name(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace("°", "")
    )


ORIENTATION_OPTIONS = {
    "vertical": "Vertical plate",
    "horizontal_free": "Horizontal, both sides open",
    "horizontal_top_open": "Horizontal, top open / underside weaker",
    "horizontal_under_blocked": "Horizontal, underside blocked",
}

ENVIRONMENT_OPTIONS = {
    "open_air": "Open air",
    "partly_enclosed": "Partly enclosed",
    "enclosed_box": "Enclosed box",
    "narrow_gap": "Narrow gap / channel",
}

SURFACE_OPTIONS = {
    "bare_metal": "Bare metal / normal",
    "matte_black": "Matte black / black anodized",
    "dark_paint": "Dark paint",
    "shiny_metal": "Shiny / polished metal",
}

AIR_MOVEMENT_OPTIONS = {
    "dead_still": "Dead still air",
    "still_air": "Normal still room air",
    "slight_movement": "Slight room movement",
    "noticeable_draft": "Noticeable natural draft",
}

HOT_AIR_PATH_OPTIONS = {
    "free_rise": "Hot air can rise freely",
    "somewhat_blocked": "Hot air partly blocked",
    "trapped_above": "Hot air trapped above plate",
}


def estimate_passive_h(
    orientation: str,
    environment: str,
    wall_clearance_cm: float,
    surface_finish: str,
    air_movement: str,
    hot_air_path: str,
    blockage_percent: float,
) -> Tuple[float, str]:
    """
    Conservative heuristic for an effective passive convection coefficient h.

    This does not replace measurement. It only helps choose a less fantasy-like h.
    The simulator still uses one uniform h across both large plate faces.
    """

    # Start from a geometry/orientation base. These values are intentionally simple.
    # They represent effective still-air cooling from the two large faces.
    orientation_h = {
        "vertical": 7.0,
        "horizontal_free": 5.8,
        "horizontal_top_open": 5.0,
        "horizontal_under_blocked": 3.8,
    }

    environment_mult = {
        "open_air": 1.00,
        "partly_enclosed": 0.70,
        "enclosed_box": 0.40,
        "narrow_gap": 0.45,
    }

    # Surface finish is mostly radiation, not pure convection. We fold it in as a small
    # equivalent-h correction so the simple model remains usable.
    surface_mult = {
        "bare_metal": 1.00,
        "matte_black": 1.10,
        "dark_paint": 1.08,
        "shiny_metal": 0.90,
    }

    air_mult = {
        "dead_still": 0.80,
        "still_air": 1.00,
        "slight_movement": 1.25,
        "noticeable_draft": 1.60,
    }

    hot_path_mult = {
        "free_rise": 1.00,
        "somewhat_blocked": 0.75,
        "trapped_above": 0.50,
    }

    c = max(0.0, wall_clearance_cm)
    if c >= 20:
        clearance_mult = 1.00
    elif c >= 10:
        clearance_mult = 0.90
    elif c >= 5:
        clearance_mult = 0.75
    elif c >= 2:
        clearance_mult = 0.55
    elif c >= 1:
        clearance_mult = 0.40
    else:
        clearance_mult = 0.25

    block = min(100.0, max(0.0, blockage_percent))
    blockage_mult = max(0.35, 1.0 - 0.006 * block)

    base_h = orientation_h.get(orientation, 7.0)
    h = base_h
    h *= environment_mult.get(environment, 1.0)
    h *= clearance_mult
    h *= surface_mult.get(surface_finish, 1.0)
    h *= air_mult.get(air_movement, 1.0)
    h *= hot_path_mult.get(hot_air_path, 1.0)
    h *= blockage_mult

    # Keep it within a sane range for passive/no-fan setups.
    h = max(1.0, min(15.0, h))

    notes = [
        f"orientation base h={base_h:.2f}",
        f"environment ×{environment_mult.get(environment, 1.0):.2f}",
        f"clearance ×{clearance_mult:.2f}",
        f"surface ×{surface_mult.get(surface_finish, 1.0):.2f}",
        f"air movement ×{air_mult.get(air_movement, 1.0):.2f}",
        f"hot-air path ×{hot_path_mult.get(hot_air_path, 1.0):.2f}",
        f"blockage ×{blockage_mult:.2f}",
    ]
    return h, "; ".join(notes)


def display_from_key(options: Dict[str, str], key: str) -> str:
    return options.get(key, key)


def key_from_display(options: Dict[str, str], display: str) -> str:
    for k, v in options.items():
        if v == display:
            return k
    return display


def build_grid(cfg: PlateConfig):
    plate_l_m = cfg.plate_length_cm / 100.0
    plate_w_m = cfg.plate_width_cm / 100.0
    dx = cfg.grid_mm / 1000.0
    dy = dx

    nx = max(3, int(round(plate_l_m / dx)))
    ny = max(3, int(round(plate_w_m / dy)))

    # Recompute so grid exactly spans the plate.
    dx = plate_l_m / nx
    dy = plate_w_m / ny

    x = (np.arange(nx) + 0.5) * dx - plate_l_m / 2.0
    y = (np.arange(ny) + 0.5) * dy - plate_w_m / 2.0

    return x, y, dx, dy


def add_heat_sources(cfg: PlateConfig, x_m: np.ndarray, y_m: np.ndarray, dx_m: float, dy_m: float):
    q = np.zeros((len(x_m), len(y_m)), dtype=float)

    X, Y = np.meshgrid(x_m, y_m, indexing="ij")
    cell_area_m2 = dx_m * dy_m
    masks: List[Tuple[Resistor, np.ndarray, float]] = []

    for r in cfg.resistors:
        cx = r.center_x_cm / 100.0
        cy = r.center_y_cm / 100.0
        half_l = (r.length_mm / 1000.0) / 2.0
        half_w = (r.width_mm / 1000.0) / 2.0

        mask = (
            (X >= cx - half_l)
            & (X <= cx + half_l)
            & (Y >= cy - half_w)
            & (Y <= cy + half_w)
        )

        # If the grid is too coarse and misses the resistor, force nearest cell.
        if not np.any(mask):
            ix = int(np.argmin(np.abs(x_m - cx)))
            iy = int(np.argmin(np.abs(y_m - cy)))
            mask[ix, iy] = True

        covered_area_m2 = float(np.count_nonzero(mask) * cell_area_m2)
        q[mask] += r.power_w / covered_area_m2
        masks.append((r, mask, covered_area_m2))

    return q, masks


def explicit_stability_dt(cfg: PlateConfig, dx_m: float, dy_m: float) -> float:
    k = cfg.thermal_conductivity_w_mk
    rho = cfg.density_kg_m3
    cp = cfg.heat_capacity_j_kgk
    t = cfg.plate_thickness_mm / 1000.0
    h = cfg.convection_h_w_m2k

    alpha = k / (rho * cp)
    beta = 2.0 * h / (rho * cp * t)

    denom = 2.0 * alpha * ((1.0 / (dx_m * dx_m)) + (1.0 / (dy_m * dy_m))) + beta
    return 0.45 / denom


def solve_steady_state(
    cfg: PlateConfig,
    q_w_m2: np.ndarray,
    dx_m: float,
    dy_m: float,
    progress_callback=None,
    stop_event: Optional[threading.Event] = None,
    max_iter: int = 30000,
    tolerance_c: float = 0.0005,
) -> Tuple[np.ndarray, Dict]:
    """
    Solves:
      k*t*laplacian(theta) + q - 2*h*theta = 0

    theta is temperature rise over ambient.
    """
    k = cfg.thermal_conductivity_w_mk
    t = cfg.plate_thickness_mm / 1000.0
    h = cfg.convection_h_w_m2k

    theta = np.zeros_like(q_w_m2, dtype=float)

    ax = k * t / (dx_m * dx_m)
    ay = k * t / (dy_m * dy_m)
    center = 2.0 * ax + 2.0 * ay + 2.0 * h

    converged = False
    final_diff = None

    last_report = time.time()
    for it in range(1, max_iter + 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Simulation was cancelled.")

        p = np.pad(theta, ((1, 1), (1, 1)), mode="edge")
        left = p[:-2, 1:-1]
        right = p[2:, 1:-1]
        down = p[1:-1, :-2]
        up = p[1:-1, 2:]

        new_theta = (ax * (left + right) + ay * (down + up) + q_w_m2) / center
        final_diff = float(np.max(np.abs(new_theta - theta)))
        theta = new_theta

        now = time.time()
        if progress_callback is not None and now - last_report > 0.2:
            progress_callback(f"Steady-state iteration {it}, change {final_diff:.5f} °C")
            last_report = now

        if final_diff < tolerance_c:
            converged = True
            break

    temp_c = cfg.ambient_c + theta
    info = {
        "iterations": it,
        "converged": converged,
        "final_diff_c": final_diff,
    }
    return temp_c, info


def run_simulation(
    cfg: PlateConfig,
    progress_callback=None,
    stop_event: Optional[threading.Event] = None,
) -> SimulationResult:
    if len(cfg.resistors) == 0:
        raise ValueError("Add at least one resistor.")

    x_m, y_m, dx_m, dy_m = build_grid(cfg)
    q_w_m2, resistor_masks = add_heat_sources(cfg, x_m, y_m, dx_m, dy_m)

    rho = cfg.density_kg_m3
    cp = cfg.heat_capacity_j_kgk
    t_m = cfg.plate_thickness_mm / 1000.0
    k = cfg.thermal_conductivity_w_mk
    h = cfg.convection_h_w_m2k

    alpha = k / (rho * cp)
    beta = 2.0 * h / (rho * cp * t_m)
    source = q_w_m2 / (rho * cp * t_m)

    dt_stable = explicit_stability_dt(cfg, dx_m, dy_m)

    max_time_s = cfg.max_time_s
    snapshot_every_s = cfg.snapshot_every_s
    snapshot_times = list(np.arange(0.0, max_time_s + 0.5 * snapshot_every_s, snapshot_every_s))
    if snapshot_times[-1] < max_time_s:
        snapshot_times.append(max_time_s)
    snapshot_times = sorted(set(float(round(v, 9)) for v in snapshot_times if v <= max_time_s + 1e-9))

    # Use an internal timestep that never skips snapshots too crudely.
    dt = min(dt_stable, snapshot_every_s / 5.0)
    steps = max(1, int(math.ceil(max_time_s / dt)))
    dt = max_time_s / steps

    estimated_cells = len(x_m) * len(y_m)
    estimated_snapshots = len(snapshot_times)

    if progress_callback is not None:
        progress_callback(
            f"Grid {len(x_m)} × {len(y_m)} = {estimated_cells} cells. "
            f"Internal timestep {dt:.4f} s. Steps {steps}. Snapshots {estimated_snapshots}."
        )

    theta = np.full_like(q_w_m2, cfg.initial_plate_temp_c - cfg.ambient_c, dtype=float)
    snapshots: List[SimulationSnapshot] = []

    snapshot_index = 0
    next_snapshot_time = snapshot_times[snapshot_index]

    def store_snapshot(t_s: float):
        snapshots.append(
            SimulationSnapshot(
                label=format_time(t_s),
                time_s=t_s,
                temp_c=(cfg.ambient_c + theta).copy(),
                is_steady_state=False,
            )
        )

    store_snapshot(0.0)
    snapshot_index = 1

    last_report = time.time()
    for step in range(1, steps + 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Simulation was cancelled.")

        p = np.pad(theta, ((1, 1), (1, 1)), mode="edge")
        left = p[:-2, 1:-1]
        right = p[2:, 1:-1]
        down = p[1:-1, :-2]
        up = p[1:-1, 2:]

        lap = (
            (left - 2.0 * theta + right) / (dx_m * dx_m)
            + (down - 2.0 * theta + up) / (dy_m * dy_m)
        )

        theta += dt * (alpha * lap + source - beta * theta)
        current_time_s = step * dt

        while snapshot_index < len(snapshot_times) and current_time_s >= snapshot_times[snapshot_index] - 0.5 * dt:
            store_snapshot(snapshot_times[snapshot_index])
            snapshot_index += 1

        now = time.time()
        if progress_callback is not None and now - last_report > 0.2:
            pct = 100.0 * current_time_s / max_time_s if max_time_s > 0 else 100.0
            progress_callback(f"Transient simulation {pct:.1f}% ({format_time(current_time_s)} / {format_time(max_time_s)})")
            last_report = now

    steady_state_temp_c = None
    steady_info = None

    if cfg.include_steady_state:
        if progress_callback is not None:
            progress_callback("Solving steady-state final heatmap...")
        steady_state_temp_c, steady_info = solve_steady_state(
            cfg,
            q_w_m2,
            dx_m,
            dy_m,
            progress_callback=progress_callback,
            stop_event=stop_event,
        )
        snapshots.append(
            SimulationSnapshot(
                label="steady-state",
                time_s=None,
                temp_c=steady_state_temp_c.copy(),
                is_steady_state=True,
            )
        )

    summary = calculate_summary(
        cfg=cfg,
        x_m=x_m,
        y_m=y_m,
        dx_m=dx_m,
        dy_m=dy_m,
        q_w_m2=q_w_m2,
        resistor_masks=resistor_masks,
        snapshots=snapshots,
        steady_info=steady_info,
    )

    return SimulationResult(
        cfg=cfg,
        x_m=x_m,
        y_m=y_m,
        dx_m=dx_m,
        dy_m=dy_m,
        q_w_m2=q_w_m2,
        resistor_masks=resistor_masks,
        snapshots=snapshots,
        steady_state_temp_c=steady_state_temp_c,
        summary=summary,
    )


def calculate_summary(
    cfg: PlateConfig,
    x_m,
    y_m,
    dx_m: float,
    dy_m: float,
    q_w_m2: np.ndarray,
    resistor_masks,
    snapshots: List[SimulationSnapshot],
    steady_info: Optional[Dict],
) -> Dict:
    plate_l_m = cfg.plate_length_cm / 100.0
    plate_w_m = cfg.plate_width_cm / 100.0
    plate_t_m = cfg.plate_thickness_mm / 1000.0
    top_bottom_area_m2 = 2.0 * plate_l_m * plate_w_m
    edge_area_m2 = 2.0 * (plate_l_m + plate_w_m) * plate_t_m
    mass_kg = plate_l_m * plate_w_m * plate_t_m * cfg.density_kg_m3
    heat_capacity_j_c = mass_kg * cfg.heat_capacity_j_kgk
    total_power_w = sum(r.power_w for r in cfg.resistors)

    simple_average_final_rise_c = total_power_w / (cfg.convection_h_w_m2k * top_bottom_area_m2)

    tau_s = heat_capacity_j_c / (cfg.convection_h_w_m2k * top_bottom_area_m2)
    t90_s = -math.log(0.10) * tau_s
    t95_s = -math.log(0.05) * tau_s

    snap_rows = []
    cell_area = dx_m * dy_m

    for snap in snapshots:
        temp = snap.temp_c
        theta = temp - cfg.ambient_c
        row = {
            "label": snap.label,
            "time_s": snap.time_s,
            "is_steady_state": snap.is_steady_state,
            "avg_temp_c": float(np.mean(temp)),
            "max_temp_c": float(np.max(temp)),
            "min_temp_c": float(np.min(temp)),
            "convective_loss_w": float(np.sum(2.0 * cfg.convection_h_w_m2k * theta * cell_area)),
        }

        for r, mask, covered_area_m2 in resistor_masks:
            row[f"{r.name}_avg_temp_c"] = float(np.mean(temp[mask]))
            row[f"{r.name}_max_temp_c"] = float(np.max(temp[mask]))

        snap_rows.append(row)

    resistor_reports = []
    final_snap = snapshots[-1]
    for r, mask, covered_area_m2 in resistor_masks:
        resistor_reports.append({
            "name": r.name,
            "power_w": r.power_w,
            "covered_area_cm2": covered_area_m2 * 10000.0,
            "final_avg_temp_c": float(np.mean(final_snap.temp_c[mask])),
            "final_max_temp_c": float(np.max(final_snap.temp_c[mask])),
        })

    return {
        "total_power_w": total_power_w,
        "effective_h_w_m2k": cfg.convection_h_w_m2k,
        "advanced_cooling_enabled": cfg.advanced_cooling_enabled,
        "cooling_notes": cfg.cooling_notes,
        "plate_area_both_faces_cm2": top_bottom_area_m2 * 10000.0,
        "plate_area_with_edges_cm2": (top_bottom_area_m2 + edge_area_m2) * 10000.0,
        "mass_kg": mass_kg,
        "heat_capacity_j_per_c": heat_capacity_j_c,
        "simple_average_final_rise_c": simple_average_final_rise_c,
        "tau_s": tau_s,
        "t90_s": t90_s,
        "t95_s": t95_s,
        "snapshots": snap_rows,
        "resistors": resistor_reports,
        "steady_state_solver": steady_info,
        "grid_cells_x": len(x_m),
        "grid_cells_y": len(y_m),
        "grid_mm_actual_x": dx_m * 1000.0,
        "grid_mm_actual_y": dy_m * 1000.0,
    }


def save_temperature_grid_csv(x_m: np.ndarray, y_m: np.ndarray, temp_c: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_cm", "y_cm", "temp_c"])
        for ix, xv in enumerate(x_m * 100.0):
            for iy, yv in enumerate(y_m * 100.0):
                writer.writerow([f"{xv:.6f}", f"{yv:.6f}", f"{temp_c[ix, iy]:.6f}"])


class ThermalPlateGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Thermal Plate Simulator v4.1")
        self.geometry("1280x820")
        self.minsize(1120, 720)

        self.resistors: List[Resistor] = [
            Resistor("R1", 50.0, 0.0, 0.0, 50.0, 20.0)
        ]

        self.result: Optional[SimulationResult] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.msg_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        # GUI redraw guards. Without these, Tk's slider callback can recursively
        # trigger itself on some platforms, especially Windows/Python 3.12.
        self._ignore_slider_callback = False
        self._drawing_snapshot = False
        self._current_snapshot_idx: Optional[int] = None

        self.vmin: Optional[float] = None
        self.vmax: Optional[float] = None

        self._build_ui()
        self._refresh_resistor_tree()
        self._update_estimated_h()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_controls(left)
        self._build_plot_area(right)

    def _build_controls(self, parent):
        row = 0

        plate_frame = ttk.LabelFrame(parent, text="Plate", padding=8)
        plate_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.plate_length_var = tk.StringVar(value="25")
        self.plate_width_var = tk.StringVar(value="25")
        self.plate_thickness_var = tk.StringVar(value="5")
        self.material_var = tk.StringVar(value="aluminium")
        self.k_var = tk.StringVar(value="205")
        self.rho_var = tk.StringVar(value="2700")
        self.cp_var = tk.StringVar(value="900")

        self._entry_row(plate_frame, 0, "Length cm", self.plate_length_var)
        self._entry_row(plate_frame, 1, "Width cm", self.plate_width_var)
        self._entry_row(plate_frame, 2, "Thickness mm", self.plate_thickness_var)

        ttk.Label(plate_frame, text="Material").grid(row=3, column=0, sticky="w", pady=2)
        mat = ttk.Combobox(
            plate_frame,
            textvariable=self.material_var,
            values=list(MATERIALS.keys()),
            state="readonly",
            width=14,
        )
        mat.grid(row=3, column=1, sticky="ew", pady=2)
        mat.bind("<<ComboboxSelected>>", self._material_changed)

        self._entry_row(plate_frame, 4, "k W/mK", self.k_var)
        self._entry_row(plate_frame, 5, "Density kg/m³", self.rho_var)
        self._entry_row(plate_frame, 6, "Heat cap J/kgK", self.cp_var)

        cooling_frame = ttk.LabelFrame(parent, text="Cooling / time", padding=8)
        cooling_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.ambient_var = tk.StringVar(value="25")
        self.h_var = tk.StringVar(value="7")
        self.grid_var = tk.StringVar(value="5")
        self.initial_temp_var = tk.StringVar(value="25")
        self.max_time_var = tk.StringVar(value="1h")
        self.snapshot_every_var = tk.StringVar(value="1m")
        self.include_steady_var = tk.BooleanVar(value=True)

        self._entry_row(cooling_frame, 0, "Ambient °C", self.ambient_var)
        self._entry_row(cooling_frame, 1, "h W/m²K", self.h_var)
        self._entry_row(cooling_frame, 2, "Grid mm", self.grid_var)
        self._entry_row(cooling_frame, 3, "Initial °C", self.initial_temp_var)
        self._entry_row(cooling_frame, 4, "Max time", self.max_time_var)
        self._entry_row(cooling_frame, 5, "Snapshot every", self.snapshot_every_var)
        ttk.Checkbutton(
            cooling_frame,
            text="Include steady-state final",
            variable=self.include_steady_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))

        hint = ttk.Label(
            cooling_frame,
            text="h: 5 weak passive, 7 normal, 10 good passive\nTimes: 30s, 1m, 15m, 1h. Plain number = minutes.",
            foreground="#555555",
            justify="left",
        )
        hint.grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))

        advanced_frame = ttk.LabelFrame(parent, text="Advanced cooling estimate", padding=8)
        advanced_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.advanced_cooling_var = tk.BooleanVar(value=False)
        self.orientation_var = tk.StringVar(value=ORIENTATION_OPTIONS["vertical"])
        self.environment_var = tk.StringVar(value=ENVIRONMENT_OPTIONS["open_air"])
        self.clearance_var = tk.StringVar(value="20")
        self.surface_var = tk.StringVar(value=SURFACE_OPTIONS["bare_metal"])
        self.air_movement_var = tk.StringVar(value=AIR_MOVEMENT_OPTIONS["still_air"])
        self.hot_air_path_var = tk.StringVar(value=HOT_AIR_PATH_OPTIONS["free_rise"])
        self.blockage_var = tk.StringVar(value="0")
        self.estimated_h_var = tk.StringVar(value="Estimated h: disabled")

        ttk.Checkbutton(
            advanced_frame,
            text="Use advanced estimate instead of manual h",
            variable=self.advanced_cooling_var,
            command=self._update_estimated_h,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self._combo_row(advanced_frame, 1, "Orientation", self.orientation_var, list(ORIENTATION_OPTIONS.values()))
        self._combo_row(advanced_frame, 2, "Environment", self.environment_var, list(ENVIRONMENT_OPTIONS.values()))
        self._entry_row(advanced_frame, 3, "Wall gap cm", self.clearance_var)
        self._combo_row(advanced_frame, 4, "Surface", self.surface_var, list(SURFACE_OPTIONS.values()))
        self._combo_row(advanced_frame, 5, "Air movement", self.air_movement_var, list(AIR_MOVEMENT_OPTIONS.values()))
        self._combo_row(advanced_frame, 6, "Hot air path", self.hot_air_path_var, list(HOT_AIR_PATH_OPTIONS.values()))
        self._entry_row(advanced_frame, 7, "Blockage %", self.blockage_var)

        ttk.Button(advanced_frame, text="Update estimate", command=self._update_estimated_h).grid(row=8, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(advanced_frame, textvariable=self.estimated_h_var, foreground="#333333").grid(row=8, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        adv_hint = ttk.Label(
            advanced_frame,
            text="This is a conservative effective-h guess. Real testing still wins.",
            foreground="#555555",
            justify="left",
        )
        adv_hint.grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))

        for var in [
            self.orientation_var, self.environment_var, self.clearance_var,
            self.surface_var, self.air_movement_var, self.hot_air_path_var,
            self.blockage_var,
        ]:
            var.trace_add("write", lambda *args: self._update_estimated_h())

        resistor_frame = ttk.LabelFrame(parent, text="Resistors", padding=8)
        resistor_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        cols = ("name", "power", "x", "y", "len", "wid")
        self.res_tree = ttk.Treeview(resistor_frame, columns=cols, show="headings", height=5)
        headings = {
            "name": "Name",
            "power": "W",
            "x": "x cm",
            "y": "y cm",
            "len": "L mm",
            "wid": "W mm",
        }
        widths = {
            "name": 58,
            "power": 50,
            "x": 50,
            "y": 50,
            "len": 55,
            "wid": 55,
        }
        for c in cols:
            self.res_tree.heading(c, text=headings[c])
            self.res_tree.column(c, width=widths[c], anchor="center")
        self.res_tree.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.res_tree.bind("<<TreeviewSelect>>", self._resistor_selected)

        self.r_name_var = tk.StringVar(value="R1")
        self.r_power_var = tk.StringVar(value="50")
        self.r_x_var = tk.StringVar(value="0")
        self.r_y_var = tk.StringVar(value="0")
        self.r_len_var = tk.StringVar(value="50")
        self.r_wid_var = tk.StringVar(value="20")

        fields = [
            ("Name", self.r_name_var),
            ("Power W", self.r_power_var),
            ("x cm", self.r_x_var),
            ("y cm", self.r_y_var),
            ("Len mm", self.r_len_var),
            ("Wid mm", self.r_wid_var),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(resistor_frame, text=label).grid(row=1 + i // 2, column=(i % 2) * 2, sticky="w", pady=2)
            ttk.Entry(resistor_frame, textvariable=var, width=10).grid(row=1 + i // 2, column=(i % 2) * 2 + 1, sticky="ew", pady=2)

        btn_frame = ttk.Frame(resistor_frame)
        btn_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(btn_frame, text="Add", command=self._add_resistor).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Update", command=self._update_resistor).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Delete", command=self._delete_resistor).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Spread 4", command=self._spread_four_example).pack(side="left")

        run_frame = ttk.LabelFrame(parent, text="Run", padding=8)
        run_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.run_button = ttk.Button(run_frame, text="Run simulation", command=self._start_simulation)
        self.run_button.grid(row=0, column=0, sticky="ew", pady=2)

        self.cancel_button = ttk.Button(run_frame, text="Cancel", command=self._cancel_simulation, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="ew", pady=2, padx=(4, 0))

        ttk.Button(run_frame, text="Save config", command=self._save_config).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(run_frame, text="Load config", command=self._load_config).grid(row=1, column=1, sticky="ew", pady=2, padx=(4, 0))
        ttk.Button(run_frame, text="Export image", command=self._export_current_image).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(run_frame, text="Export CSV", command=self._export_current_csv).grid(row=2, column=1, sticky="ew", pady=2, padx=(4, 0))

        self.fixed_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            run_frame,
            text="Fixed color scale across snapshots",
            variable=self.fixed_scale_var,
            command=self._redraw_current_snapshot,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        status_frame = ttk.LabelFrame(parent, text="Status / summary", padding=8)
        status_frame.grid(row=row, column=0, sticky="nsew")
        parent.rowconfigure(row, weight=1)

        self.status_text = tk.Text(status_frame, width=42, height=14, wrap="word")
        self.status_text.grid(row=0, column=0, sticky="nsew")
        status_scroll = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        status_scroll.grid(row=0, column=1, sticky="ns")
        self.status_text.configure(yscrollcommand=status_scroll.set)

        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self._set_status("Ready. Enter values and press Run simulation.")

    def _build_plot_area(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No simulation yet")
        self.ax.set_xlabel("x cm")
        self.ax.set_ylabel("y cm")
        self.colorbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        slider_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        slider_frame.grid(row=1, column=0, sticky="ew")
        slider_frame.columnconfigure(1, weight=1)

        ttk.Label(slider_frame, text="Time").grid(row=0, column=0, sticky="w", padx=(0, 6))

        self.slider_var = tk.IntVar(value=0)
        self.time_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._slider_changed,
        )
        self.time_slider.grid(row=0, column=1, sticky="ew")

        self.snapshot_label_var = tk.StringVar(value="No snapshots")
        ttk.Label(slider_frame, textvariable=self.snapshot_label_var, width=24).grid(row=0, column=2, sticky="e", padx=(8, 0))

        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=2, column=0, sticky="ew")
        ttk.Button(nav_frame, text="Previous", command=lambda: self._step_slider(-1)).pack(side="left")
        ttk.Button(nav_frame, text="Next", command=lambda: self._step_slider(1)).pack(side="left", padx=(4, 0))

    def _entry_row(self, parent, row: int, label: str, var: tk.StringVar):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=14)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)
        return entry

    def _combo_row(self, parent, row: int, label: str, var: tk.StringVar, values: List[str]):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=24)
        combo.grid(row=row, column=1, sticky="ew", pady=2)
        combo.bind("<<ComboboxSelected>>", lambda event: self._update_estimated_h())
        parent.columnconfigure(1, weight=1)
        return combo

    def _material_changed(self, event=None):
        name = self.material_var.get()
        if name in MATERIALS:
            m = MATERIALS[name]
            self.k_var.set(str(m["k"]))
            self.rho_var.set(str(m["rho"]))
            self.cp_var.set(str(m["cp"]))

    def _refresh_resistor_tree(self):
        for item in self.res_tree.get_children():
            self.res_tree.delete(item)

        for i, r in enumerate(self.resistors):
            self.res_tree.insert(
                "",
                "end",
                iid=str(i),
                values=(
                    r.name,
                    f"{r.power_w:g}",
                    f"{r.center_x_cm:g}",
                    f"{r.center_y_cm:g}",
                    f"{r.length_mm:g}",
                    f"{r.width_mm:g}",
                ),
            )

    def _resistor_selected(self, event=None):
        sel = self.res_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.resistors):
            r = self.resistors[idx]
            self.r_name_var.set(r.name)
            self.r_power_var.set(f"{r.power_w:g}")
            self.r_x_var.set(f"{r.center_x_cm:g}")
            self.r_y_var.set(f"{r.center_y_cm:g}")
            self.r_len_var.set(f"{r.length_mm:g}")
            self.r_wid_var.set(f"{r.width_mm:g}")

    def _read_resistor_fields(self) -> Resistor:
        name = self.r_name_var.get().strip() or f"R{len(self.resistors)+1}"
        power = parse_float(self.r_power_var.get(), "Resistor power", 0.0)
        x = parse_float(self.r_x_var.get(), "Resistor x")
        y = parse_float(self.r_y_var.get(), "Resistor y")
        length = parse_float(self.r_len_var.get(), "Resistor length", 0.1)
        width = parse_float(self.r_wid_var.get(), "Resistor width", 0.1)
        return Resistor(name, power, x, y, length, width)

    def _add_resistor(self):
        try:
            self.resistors.append(self._read_resistor_fields())
            self._refresh_resistor_tree()
        except Exception as e:
            messagebox.showerror("Invalid resistor", str(e))

    def _update_resistor(self):
        sel = self.res_tree.selection()
        if not sel:
            messagebox.showinfo("Update resistor", "Select a resistor first.")
            return
        try:
            idx = int(sel[0])
            self.resistors[idx] = self._read_resistor_fields()
            self._refresh_resistor_tree()
            self.res_tree.selection_set(str(idx))
        except Exception as e:
            messagebox.showerror("Invalid resistor", str(e))

    def _delete_resistor(self):
        sel = self.res_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.resistors):
            del self.resistors[idx]
            self._refresh_resistor_tree()

    def _spread_four_example(self):
        self.resistors = [
            Resistor("R1", 50.0, -5.0, -5.0, 50.0, 20.0),
            Resistor("R2", 50.0,  5.0, -5.0, 50.0, 20.0),
            Resistor("R3", 50.0, -5.0,  5.0, 50.0, 20.0),
            Resistor("R4", 50.0,  5.0,  5.0, 50.0, 20.0),
        ]
        self._refresh_resistor_tree()
        self._set_status("Loaded example with four 50 W resistors spread around the center.")

    def _estimate_h_from_ui(self) -> Tuple[float, str]:
        orientation = key_from_display(ORIENTATION_OPTIONS, self.orientation_var.get())
        environment = key_from_display(ENVIRONMENT_OPTIONS, self.environment_var.get())
        surface = key_from_display(SURFACE_OPTIONS, self.surface_var.get())
        air = key_from_display(AIR_MOVEMENT_OPTIONS, self.air_movement_var.get())
        hot_path = key_from_display(HOT_AIR_PATH_OPTIONS, self.hot_air_path_var.get())
        clearance = parse_float(self.clearance_var.get(), "Wall gap", 0.0)
        blockage = parse_float(self.blockage_var.get(), "Blockage", 0.0)
        return estimate_passive_h(
            orientation=orientation,
            environment=environment,
            wall_clearance_cm=clearance,
            surface_finish=surface,
            air_movement=air,
            hot_air_path=hot_path,
            blockage_percent=blockage,
        )

    def _update_estimated_h(self):
        try:
            h, _notes = self._estimate_h_from_ui()
            if self.advanced_cooling_var.get():
                self.estimated_h_var.set(f"Using h ≈ {h:.2f} W/m²K")
            else:
                self.estimated_h_var.set(f"Estimate h ≈ {h:.2f} W/m²K")
        except Exception:
            if hasattr(self, "estimated_h_var"):
                self.estimated_h_var.set("Estimate h: invalid advanced input")

    def _read_config(self) -> PlateConfig:
        advanced_enabled = bool(self.advanced_cooling_var.get())
        estimated_h, cooling_notes = self._estimate_h_from_ui()
        manual_h = parse_float(self.h_var.get(), "Convection h", 0.1)
        effective_h = estimated_h if advanced_enabled else manual_h

        cfg = PlateConfig(
            plate_length_cm=parse_float(self.plate_length_var.get(), "Plate length", 0.1),
            plate_width_cm=parse_float(self.plate_width_var.get(), "Plate width", 0.1),
            plate_thickness_mm=parse_float(self.plate_thickness_var.get(), "Plate thickness", 0.1),
            material_name=self.material_var.get(),
            thermal_conductivity_w_mk=parse_float(self.k_var.get(), "Thermal conductivity k", 0.1),
            density_kg_m3=parse_float(self.rho_var.get(), "Density", 1.0),
            heat_capacity_j_kgk=parse_float(self.cp_var.get(), "Heat capacity", 1.0),
            ambient_c=parse_float(self.ambient_var.get(), "Ambient temperature"),
            convection_h_w_m2k=effective_h,
            grid_mm=parse_float(self.grid_var.get(), "Grid size", 0.5),
            resistors=list(self.resistors),
            initial_plate_temp_c=parse_float(self.initial_temp_var.get(), "Initial plate temperature"),
            max_time_s=parse_time_to_seconds(self.max_time_var.get(), "Max time"),
            snapshot_every_s=parse_time_to_seconds(self.snapshot_every_var.get(), "Snapshot every"),
            include_steady_state=bool(self.include_steady_var.get()),
            advanced_cooling_enabled=advanced_enabled,
            orientation=key_from_display(ORIENTATION_OPTIONS, self.orientation_var.get()),
            environment=key_from_display(ENVIRONMENT_OPTIONS, self.environment_var.get()),
            wall_clearance_cm=parse_float(self.clearance_var.get(), "Wall gap", 0.0),
            surface_finish=key_from_display(SURFACE_OPTIONS, self.surface_var.get()),
            air_movement=key_from_display(AIR_MOVEMENT_OPTIONS, self.air_movement_var.get()),
            hot_air_path=key_from_display(HOT_AIR_PATH_OPTIONS, self.hot_air_path_var.get()),
            blockage_percent=parse_float(self.blockage_var.get(), "Blockage", 0.0),
            cooling_notes=cooling_notes,
        )

        if cfg.snapshot_every_s > cfg.max_time_s:
            raise ValueError("Snapshot interval cannot be greater than max time.")

        if len(cfg.resistors) == 0:
            raise ValueError("Add at least one resistor.")

        # Guard against absurd amount of snapshots.
        snapshot_count = int(math.floor(cfg.max_time_s / cfg.snapshot_every_s)) + 1
        if snapshot_count > 500:
            raise ValueError(
                f"This would create about {snapshot_count} snapshots. "
                "Increase 'Snapshot every' or reduce 'Max time'."
            )

        # Guard against huge grid.
        nx = max(3, int(round((cfg.plate_length_cm / 100.0) / (cfg.grid_mm / 1000.0))))
        ny = max(3, int(round((cfg.plate_width_cm / 100.0) / (cfg.grid_mm / 1000.0))))
        if nx * ny > 200_000:
            raise ValueError(
                f"Grid would be {nx} × {ny} = {nx*ny} cells. "
                "Use a larger grid size, e.g. 5 mm or 10 mm."
            )

        return cfg

    def _start_simulation(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Running", "A simulation is already running.")
            return

        try:
            cfg = self._read_config()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.result = None
        self.vmin = None
        self.vmax = None
        self.stop_event.clear()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._set_status("Starting simulation...")

        def worker():
            try:
                def progress(msg: str):
                    self.msg_queue.put(("progress", msg))

                result = run_simulation(cfg, progress_callback=progress, stop_event=self.stop_event)
                self.msg_queue.put(("done", result))
            except Exception as e:
                tb = traceback.format_exc()
                self.msg_queue.put(("error", f"{e}\n\n{tb}"))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _cancel_simulation(self):
        self.stop_event.set()
        self._append_status("Cancelling...")

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()

                if msg_type == "progress":
                    self._set_progress_line(str(payload))
                elif msg_type == "done":
                    self._simulation_done(payload)
                elif msg_type == "error":
                    self.run_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self._append_status("Error:\n" + str(payload))
                    messagebox.showerror("Simulation error", str(payload).split("\n\n")[0])
        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    def _simulation_done(self, result: SimulationResult):
        self.result = result
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

        all_temps = np.concatenate([s.temp_c.ravel() for s in result.snapshots])
        self.vmin = float(np.min(all_temps))
        self.vmax = float(np.max(all_temps))

        max_index = max(0, len(result.snapshots) - 1)
        self.time_slider.configure(from_=0, to=max_index)
        self._ignore_slider_callback = True
        try:
            self.time_slider.set(0)
        finally:
            self._ignore_slider_callback = False
        self._current_snapshot_idx = None

        self._write_summary(result)
        self._draw_snapshot(0)

    def _write_summary(self, result: SimulationResult):
        s = result.summary
        lines = []
        lines.append("Simulation complete.")
        lines.append("")
        lines.append(f"Grid: {s['grid_cells_x']} × {s['grid_cells_y']}")
        lines.append(f"Total power: {s['total_power_w']:.2f} W")
        lines.append(f"Effective h used: {result.cfg.convection_h_w_m2k:.2f} W/m²K")
        if result.cfg.advanced_cooling_enabled:
            lines.append("Cooling mode: advanced estimate")
            lines.append(f"Cooling factors: {result.cfg.cooling_notes}")
        else:
            lines.append("Cooling mode: manual h")
        lines.append(f"Plate area, both faces: {s['plate_area_both_faces_cm2']:.0f} cm²")
        lines.append(f"Mass: {s['mass_kg']:.3f} kg")
        lines.append(f"Heat capacity: {s['heat_capacity_j_per_c']:.0f} J/°C")
        lines.append("")
        lines.append("Rough whole-plate warm-up:")
        lines.append(f"Time constant: {format_time(s['tau_s'])}")
        lines.append(f"~90% final average: {format_time(s['t90_s'])}")
        lines.append(f"~95% final average: {format_time(s['t95_s'])}")
        lines.append("")
        lines.append("Snapshots:")
        for row in s["snapshots"]:
            lines.append(
                f"  {row['label']}: avg {row['avg_temp_c']:.1f} °C, "
                f"max {row['max_temp_c']:.1f} °C"
            )
        lines.append("")
        lines.append("Resistor footprint temps at final selected endpoint:")
        for r in s["resistors"]:
            lines.append(
                f"  {r['name']}: avg {r['final_avg_temp_c']:.1f} °C, "
                f"max {r['final_max_temp_c']:.1f} °C"
            )
        lines.append("")
        lines.append("Note: resistor body/internal temp may be hotter than plate contact temp.")

        self._set_status("\n".join(lines))

    def _slider_changed(self, value):
        """
        Tk calls this continuously while the slider moves.

        Important:
        Do NOT call self.time_slider.set(...) from here. On some Tk builds,
        setting the slider inside its own callback recursively fires the callback
        again and can crash with RecursionError/TclError.
        """
        if self._ignore_slider_callback or self._drawing_snapshot:
            return
        if self.result is None:
            return

        try:
            idx = int(round(float(value)))
        except Exception:
            return

        idx = max(0, min(len(self.result.snapshots) - 1, idx))

        # Avoid redrawing the same snapshot again and again while Tk sends
        # repeated slider events.
        if self._current_snapshot_idx == idx:
            return

        self._draw_snapshot(idx)

    def _step_slider(self, delta: int):
        if self.result is None:
            return

        cur = int(round(float(self.time_slider.get())))
        new = max(0, min(len(self.result.snapshots) - 1, cur + delta))

        self._ignore_slider_callback = True
        try:
            self.time_slider.set(new)
        finally:
            self._ignore_slider_callback = False

        self._draw_snapshot(new)

    def _redraw_current_snapshot(self):
        if self.result is None:
            return
        idx = int(round(float(self.time_slider.get())))
        self._current_snapshot_idx = None
        self._draw_snapshot(idx)

    def _draw_snapshot(self, idx: int):
        if self.result is None:
            return
        if idx < 0 or idx >= len(self.result.snapshots):
            return
        if self._drawing_snapshot:
            return

        self._drawing_snapshot = True
        try:
            snap = self.result.snapshots[idx]
            temp = snap.temp_c
            cfg = self.result.cfg
            x_cm = self.result.x_m * 100.0
            y_cm = self.result.y_m * 100.0

            # Fully rebuild the figure each time. This is a little less elegant
            # than reusing axes, but it avoids Matplotlib colorbar/axis buildup
            # and is much more stable on Windows.
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.colorbar = None

            extent = [x_cm[0], x_cm[-1], y_cm[0], y_cm[-1]]

            if self.fixed_scale_var.get() and self.vmin is not None and self.vmax is not None:
                vmin = self.vmin
                vmax = self.vmax
            else:
                vmin = None
                vmax = None

            im = self.ax.imshow(
                temp.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="bilinear",
                vmin=vmin,
                vmax=vmax,
            )

            self.colorbar = self.fig.colorbar(im, ax=self.ax)
            self.colorbar.set_label("Temperature, °C")

            for r in cfg.resistors:
                cx = r.center_x_cm
                cy = r.center_y_cm
                l_cm = r.length_mm / 10.0
                w_cm = r.width_mm / 10.0
                rect = Rectangle(
                    (cx - l_cm / 2.0, cy - w_cm / 2.0),
                    l_cm,
                    w_cm,
                    fill=False,
                    linewidth=2,
                )
                self.ax.add_patch(rect)
                self.ax.text(cx, cy, r.name, ha="center", va="center", fontsize=9)

            max_idx = np.unravel_index(np.argmax(temp), temp.shape)
            max_x = x_cm[max_idx[0]]
            max_y = y_cm[max_idx[1]]
            max_t = float(temp[max_idx])
            avg_t = float(np.mean(temp))

            self.ax.plot([max_x], [max_y], marker="x", markersize=10)
            self.ax.text(max_x, max_y, f" max {max_t:.1f}°C", va="bottom")

            self.ax.set_title(f"{snap.label} | avg {avg_t:.1f}°C | max {max_t:.1f}°C")
            self.ax.set_xlabel("x position, cm")
            self.ax.set_ylabel("y position, cm")
            self.ax.grid(True, alpha=0.25)

            # Avoid tight_layout here. On some Windows/Matplotlib combinations
            # it can trigger deep recursion during rapid redraws.
            self.fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.92)
            self.canvas.draw_idle()

            self.snapshot_label_var.set(f"{idx + 1}/{len(self.result.snapshots)}: {snap.label}")
            self._current_snapshot_idx = idx
        finally:
            self._drawing_snapshot = False

    def _set_status(self, text: str):
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("end", text)
        self.status_text.configure(state="disabled")

    def _append_status(self, text: str):
        self.status_text.configure(state="normal")
        self.status_text.insert("end", "\n" + text)
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def _set_progress_line(self, text: str):
        # Keep progress compact; don't spam the box endlessly.
        current = self.status_text.get("1.0", "end").strip()
        lines = current.splitlines() if current else []
        if lines and lines[-1].startswith("Progress:"):
            lines[-1] = "Progress: " + text
        else:
            lines.append("Progress: " + text)
        self._set_status("\n".join(lines))

    def _save_config(self):
        try:
            cfg = self._read_config()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        path = filedialog.asksaveasfilename(
            title="Save config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        data = asdict(cfg)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._append_status(f"Saved config: {path}")

    def _load_config(self):
        path = filedialog.askopenfilename(
            title="Load config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            resistors = [Resistor(**r) for r in data["resistors"]]
            data["resistors"] = resistors
            cfg = PlateConfig(**data)
            self._apply_config(cfg)
            self._append_status(f"Loaded config: {path}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _apply_config(self, cfg: PlateConfig):
        self.plate_length_var.set(f"{cfg.plate_length_cm:g}")
        self.plate_width_var.set(f"{cfg.plate_width_cm:g}")
        self.plate_thickness_var.set(f"{cfg.plate_thickness_mm:g}")
        self.material_var.set(cfg.material_name)
        self.k_var.set(f"{cfg.thermal_conductivity_w_mk:g}")
        self.rho_var.set(f"{cfg.density_kg_m3:g}")
        self.cp_var.set(f"{cfg.heat_capacity_j_kgk:g}")
        self.ambient_var.set(f"{cfg.ambient_c:g}")
        self.h_var.set(f"{cfg.convection_h_w_m2k:g}")
        self.grid_var.set(f"{cfg.grid_mm:g}")
        self.initial_temp_var.set(f"{cfg.initial_plate_temp_c:g}")
        self.max_time_var.set(format_time(cfg.max_time_s).replace(" ", ""))
        self.snapshot_every_var.set(format_time(cfg.snapshot_every_s).replace(" ", ""))
        self.include_steady_var.set(cfg.include_steady_state)
        self.advanced_cooling_var.set(getattr(cfg, "advanced_cooling_enabled", False))
        self.orientation_var.set(display_from_key(ORIENTATION_OPTIONS, getattr(cfg, "orientation", "vertical")))
        self.environment_var.set(display_from_key(ENVIRONMENT_OPTIONS, getattr(cfg, "environment", "open_air")))
        self.clearance_var.set(f"{getattr(cfg, 'wall_clearance_cm', 20.0):g}")
        self.surface_var.set(display_from_key(SURFACE_OPTIONS, getattr(cfg, "surface_finish", "bare_metal")))
        self.air_movement_var.set(display_from_key(AIR_MOVEMENT_OPTIONS, getattr(cfg, "air_movement", "still_air")))
        self.hot_air_path_var.set(display_from_key(HOT_AIR_PATH_OPTIONS, getattr(cfg, "hot_air_path", "free_rise")))
        self.blockage_var.set(f"{getattr(cfg, 'blockage_percent', 0.0):g}")
        self._update_estimated_h()
        self.resistors = list(cfg.resistors)
        self._refresh_resistor_tree()

    def _get_current_snapshot(self) -> Optional[SimulationSnapshot]:
        if self.result is None:
            return None
        idx = int(round(float(self.time_slider.get())))
        if 0 <= idx < len(self.result.snapshots):
            return self.result.snapshots[idx]
        return None

    def _export_current_image(self):
        if self.result is None:
            messagebox.showinfo("No result", "Run a simulation first.")
            return

        snap = self._get_current_snapshot()
        default_name = "heatmap.png"
        if snap is not None:
            default_name = f"heatmap_{safe_time_name(snap.label)}.png"

        path = filedialog.asksaveasfilename(
            title="Export current heatmap",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        self.fig.savefig(path, dpi=180)
        self._append_status(f"Exported image: {path}")

    def _export_current_csv(self):
        if self.result is None:
            messagebox.showinfo("No result", "Run a simulation first.")
            return

        snap = self._get_current_snapshot()
        if snap is None:
            return

        default_name = f"temperature_grid_{safe_time_name(snap.label)}.csv"
        path = filedialog.asksaveasfilename(
            title="Export current temperature grid",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        save_temperature_grid_csv(self.result.x_m, self.result.y_m, snap.temp_c, Path(path))
        self._append_status(f"Exported CSV: {path}")


def main():
    app = ThermalPlateGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
