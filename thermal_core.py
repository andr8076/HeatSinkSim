#!/usr/bin/env python3
"""
thermal_plate_sim_v15_2_gui.py

Desktop GUI for simulating passive heat spreading in a flat metal plate.

Adds live layout preview, advanced cooling, geometry heatsink builder, segmented fin transfer, even-spread banks, and a deeper multi-worker optimizer.

Features:
  - Tkinter UI
  - Multiple rectangular resistors / heat sources
  - Time-based transient simulation
  - Slider through time snapshots
  - Optional steady-state final heatmap
  - Advanced cooling estimator for orientation, enclosure, wall clearance, surface, air movement, hot-air path, and blockage
  - Live plate/resistor layout preview before running
  - Even-spread resistor bank generator
  - Deep multi-worker optimizer for resistor locations using fast scoring plus coarse finite-difference heat solves
  - Field help window
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
  python3 thermal_plate_sim_v15_2_gui.py
"""

from __future__ import annotations

import csv
import concurrent.futures
import itertools
import os
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

    # Optional heatsink / fins support. This is modeled as extra effective exposed
    # area, spread across the plate. It is most accurate for a heatsink/finned
    # plate attached broadly to the back of the plate, not a tiny local heatsink.
    heatsink_enabled: bool = False
    heatsink_extra_area_cm2: float = 0.0
    heatsink_efficiency_percent: float = 70.0
    heatsink_h_multiplier: float = 1.0
    heatsink_notes: str = ""

    # Geometry heatsink builder. This models fins as actual fin strips placed
    # on the back side of the base plate. Resistors are assumed to be mounted
    # on the flat/front side.
    #
    # Orientation:
    #   run_y = fins run along plate Y/height, positions spread across X/width
    #   run_x = fins run along plate X/width, positions spread across Y/height
    heatsink_geometry_enabled: bool = False
    heatsink_fin_orientation: str = "run_y"
    heatsink_fin_count: int = 0
    heatsink_fin_thickness_mm: float = 0.0  # 0 = same as base plate thickness
    heatsink_fin_default_height_mm: float = 30.0
    heatsink_fin_run_length_cm: float = 0.0  # 0 = full available plate dimension
    heatsink_fin_positions_cm: str = "even"  # "even" or comma-separated centers
    heatsink_fin_heights_mm: str = "same"    # "same" or comma-separated heights
    heatsink_fin_segments: int = 0            # 0 = auto; split each fin along its run length for local cooling resolution

    # Resistor thermal model.
    # The solver calculates plate/contact temperature. These values estimate
    # the resistor case/body and internal element above the local plate contact.
    resistor_case_to_plate_cw: float = 0.5
    resistor_element_to_case_cw: float = 0.6


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



def _parse_number_list(text: str) -> List[float]:
    raw = str(text or "").strip().lower()
    if raw in ("", "even", "same", "auto"):
        return []
    out: List[float] = []
    for part in re.split(r"[;,\n]+", raw):
        part = part.strip().replace(",", ".")
        if not part:
            continue
        out.append(float(part))
    return out


def _value_or_default(value: float, default: float) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return default if value <= 0 else value


def heatsink_fin_specs(cfg: "PlateConfig") -> List[Dict]:
    """Return calculated fin geometry from the heatsink builder settings.

    Each fin is a rectangular vertical strip attached to the back of the base
    plate. The dimensions define the geometry; the only environmental input is
    the same convection h used for the rest of the plate.
    """
    if not getattr(cfg, "heatsink_enabled", False):
        return []
    if not getattr(cfg, "heatsink_geometry_enabled", False):
        return []

    count = int(max(0, round(float(getattr(cfg, "heatsink_fin_count", 0) or 0))))
    if count <= 0:
        return []

    orientation = str(getattr(cfg, "heatsink_fin_orientation", "run_y") or "run_y").lower()
    if orientation not in ("run_y", "run_x"):
        orientation = "run_y"

    plate_x_cm = max(0.001, float(cfg.plate_length_cm))
    plate_y_cm = max(0.001, float(cfg.plate_width_cm))
    base_t_mm = max(0.001, float(cfg.plate_thickness_mm))
    fin_t_mm = _value_or_default(getattr(cfg, "heatsink_fin_thickness_mm", 0.0), base_t_mm)
    default_h_mm = max(0.001, float(getattr(cfg, "heatsink_fin_default_height_mm", 30.0) or 30.0))

    requested_run_cm = float(getattr(cfg, "heatsink_fin_run_length_cm", 0.0) or 0.0)
    full_run_cm = plate_y_cm if orientation == "run_y" else plate_x_cm
    run_len_cm = full_run_cm if requested_run_cm <= 0 else min(full_run_cm, requested_run_cm)

    heights = _parse_number_list(getattr(cfg, "heatsink_fin_heights_mm", "same"))
    if not heights:
        heights = [default_h_mm] * count
    elif len(heights) < count:
        heights = heights + [heights[-1]] * (count - len(heights))
    else:
        heights = heights[:count]
    heights = [max(0.001, float(h)) for h in heights]

    positions = _parse_number_list(getattr(cfg, "heatsink_fin_positions_cm", "even"))
    across_cm = plate_x_cm if orientation == "run_y" else plate_y_cm
    half_fin_t_cm = fin_t_mm / 20.0
    min_pos = -across_cm / 2.0 + half_fin_t_cm
    max_pos = across_cm / 2.0 - half_fin_t_cm

    if not positions:
        if count == 1:
            positions = [0.0]
        else:
            positions = list(np.linspace(min_pos, max_pos, count))
    elif len(positions) < count:
        remaining = count - len(positions)
        fillers = [0.0] if remaining == 1 else list(np.linspace(min_pos, max_pos, remaining))
        positions = positions + fillers
    else:
        positions = positions[:count]
    positions = [min(max_pos, max(min_pos, float(x))) for x in positions]

    specs: List[Dict] = []
    for i in range(count):
        pos = positions[i]
        height_mm = heights[i]
        if orientation == "run_y":
            x_cm = pos
            y_cm = 0.0
            footprint_x_cm = fin_t_mm / 10.0
            footprint_y_cm = run_len_cm
        else:
            x_cm = 0.0
            y_cm = pos
            footprint_x_cm = run_len_cm
            footprint_y_cm = fin_t_mm / 10.0

        run_m = run_len_cm / 100.0
        thick_m = fin_t_mm / 1000.0
        height_m = height_mm / 1000.0
        # Exposed fin area: two broad faces + two end faces + tip.
        # The base contact face is not exposed.
        raw_area_m2 = 2.0 * run_m * height_m + 2.0 * thick_m * height_m + run_m * thick_m
        footprint_m2 = run_m * thick_m

        specs.append({
            "index": i + 1,
            "orientation": orientation,
            "center_x_cm": x_cm,
            "center_y_cm": y_cm,
            "position_cm": pos,
            "run_length_cm": run_len_cm,
            "thickness_mm": fin_t_mm,
            "height_mm": height_mm,
            "footprint_x_cm": footprint_x_cm,
            "footprint_y_cm": footprint_y_cm,
            "raw_area_m2": raw_area_m2,
            "footprint_m2": footprint_m2,
        })

    return specs


def fin_efficiency_for_spec(cfg: "PlateConfig", spec: Dict) -> float:
    """Classic straight rectangular fin efficiency.

    eta = tanh(m*Lc)/(m*Lc)
    m = sqrt(h*P/(k*A_c))

    This removes manual fin-efficiency guessing. The remaining environmental
    uncertainty is the convection h value.
    """
    h = max(0.05, float(cfg.convection_h_w_m2k))
    k = max(0.1, float(cfg.thermal_conductivity_w_mk))
    run_m = max(1e-9, spec["run_length_cm"] / 100.0)
    thick_m = max(1e-9, spec["thickness_mm"] / 1000.0)
    height_m = max(1e-9, spec["height_mm"] / 1000.0)

    area_c = thick_m * run_m
    perimeter = 2.0 * (run_m + thick_m)
    m = math.sqrt(h * perimeter / (k * area_c))
    corrected_length = height_m + thick_m / 2.0
    ml = m * corrected_length
    if ml <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, math.tanh(ml) / ml))



def fin_temperature_at_height(cfg: "PlateConfig", spec: Dict, base_temp_c: float, height_fraction: float) -> float:
    """Estimate fin temperature at a height fraction from base to tip.

    height_fraction:
        0.0 = attached base of fin
        1.0 = fin tip

    This uses the same straight rectangular fin theory used for fin efficiency.
    It is not full CFD, but it gives a physically meaningful gradient up the fin
    instead of coloring the whole fin as one temperature.
    """
    frac = max(0.0, min(1.0, float(height_fraction)))
    ambient = float(getattr(cfg, "ambient_c", 25.0))
    theta_b = float(base_temp_c) - ambient
    if abs(theta_b) < 1e-12:
        return float(base_temp_c)

    h = max(0.05, float(cfg.convection_h_w_m2k))
    k = max(0.1, float(cfg.thermal_conductivity_w_mk))
    run_m = max(1e-9, spec["run_length_cm"] / 100.0)
    thick_m = max(1e-9, spec["thickness_mm"] / 1000.0)
    height_m = max(1e-9, spec["height_mm"] / 1000.0)

    area_c = thick_m * run_m
    perimeter = 2.0 * (run_m + thick_m)
    m = math.sqrt(h * perimeter / (k * area_c))

    # Corrected length approximates convection from the fin tip.
    corrected_length = height_m + thick_m / 2.0
    x = frac * height_m

    denom = math.cosh(m * corrected_length)
    if denom <= 1e-12:
        return float(base_temp_c)

    ratio = math.cosh(m * max(0.0, corrected_length - x)) / denom
    return ambient + theta_b * ratio


def resistor_temperature_estimate(cfg: "PlateConfig", r: Resistor, temp_c: np.ndarray, mask: np.ndarray, covered_area_m2: float) -> Dict:
    """Estimate plate footprint, resistor case/body, and element temperatures."""
    plate_avg = float(np.mean(temp_c[mask]))
    plate_max = float(np.max(temp_c[mask]))
    case_to_plate = max(0.0, float(getattr(cfg, "resistor_case_to_plate_cw", 0.5)))
    element_to_case = max(0.0, float(getattr(cfg, "resistor_element_to_case_cw", 0.6)))

    case_avg = plate_avg + r.power_w * case_to_plate
    case_max = plate_max + r.power_w * case_to_plate
    element_avg = case_avg + r.power_w * element_to_case
    element_max = case_max + r.power_w * element_to_case

    return {
        "name": r.name,
        "power_w": float(r.power_w),
        "covered_area_cm2": covered_area_m2 * 10000.0,
        "plate_footprint_avg_temp_c": plate_avg,
        "plate_footprint_max_temp_c": plate_max,
        "resistor_case_to_plate_cw": case_to_plate,
        "resistor_element_to_case_cw": element_to_case,
        "estimated_case_avg_temp_c": case_avg,
        "estimated_case_max_temp_c": case_max,
        "estimated_element_avg_temp_c": element_avg,
        "estimated_element_max_temp_c": element_max,
    }

def _auto_fin_segment_count(cfg: "PlateConfig", spec: Dict, dx_m=None, dy_m=None) -> int:
    """Choose a sensible number of local thermal segments for one fin."""
    requested = int(max(0, round(float(getattr(cfg, "heatsink_fin_segments", 0) or 0))))
    if requested > 0:
        return max(1, min(240, requested))

    run_cm = max(0.001, float(spec["run_length_cm"]))
    orientation = spec.get("orientation", "run_y")

    if dx_m is not None and dy_m is not None:
        run_cell_cm = (dy_m if orientation == "run_y" else dx_m) * 100.0
        # v15: slightly coarser default than v14. It is much faster and still
        # usually finer than the thermal grid itself.
        target_cm = max(1.2, min(5.0, run_cell_cm * 3.0))
    else:
        target_cm = 3.0

    return max(1, min(80, int(math.ceil(run_cm / target_cm))))


def heatsink_fin_segment_specs(cfg: "PlateConfig", dx_m=None, dy_m=None) -> List[Dict]:
    """Split every fin into local run-length segments.

    v12 improvement:
    Older versions treated a full-length fin as one large cooling footprint.
    That was too coarse when a fin crossed both hot and cool parts of the plate.
    This distributes the same total fin conductance over smaller local segments,
    giving a better heat-transfer map and a better 3D visualization.
    """
    segments: List[Dict] = []

    for spec in heatsink_fin_specs(cfg):
        n = _auto_fin_segment_count(cfg, spec, dx_m=dx_m, dy_m=dy_m)
        n = max(1, int(n))

        orientation = spec.get("orientation", "run_y")
        run_len_cm = float(spec["run_length_cm"])
        seg_run_cm = run_len_cm / n
        eta = fin_efficiency_for_spec(cfg, spec)

        for j in range(n):
            offset_cm = -run_len_cm / 2.0 + seg_run_cm * (j + 0.5)
            seg = dict(spec)
            seg["parent_index"] = spec.get("index")
            seg["segment_index"] = j + 1
            seg["segment_count"] = n
            seg["full_fin_efficiency"] = eta
            seg["run_length_cm"] = seg_run_cm
            seg["raw_area_m2"] = spec["raw_area_m2"] / n
            seg["footprint_m2"] = spec["footprint_m2"] / n

            if orientation == "run_y":
                seg["center_x_cm"] = spec["center_x_cm"]
                seg["center_y_cm"] = spec["center_y_cm"] + offset_cm
                seg["footprint_x_cm"] = spec["footprint_x_cm"]
                seg["footprint_y_cm"] = seg_run_cm
            else:
                seg["center_x_cm"] = spec["center_x_cm"] + offset_cm
                seg["center_y_cm"] = spec["center_y_cm"]
                seg["footprint_x_cm"] = seg_run_cm
                seg["footprint_y_cm"] = spec["footprint_y_cm"]

            segments.append(seg)

    return segments


def heatsink_geometry_summary(cfg: "PlateConfig") -> Dict:
    specs = heatsink_fin_specs(cfg)
    raw_area_cm2 = 0.0
    effective_fin_area_cm2 = 0.0
    footprint_cm2 = 0.0
    effective_extra_cm2 = 0.0
    effs = []

    for spec in specs:
        eta = fin_efficiency_for_spec(cfg, spec)
        raw = spec["raw_area_m2"] * 10000.0
        foot = spec["footprint_m2"] * 10000.0
        raw_area_cm2 += raw
        footprint_cm2 += foot
        effective_fin_area_cm2 += eta * raw
        effective_extra_cm2 += max(0.0, eta * raw - foot)
        effs.append(eta)

    return {
        "enabled": bool(specs),
        "fin_count": len(specs),
        "raw_fin_area_cm2": raw_area_cm2,
        "effective_fin_area_cm2": effective_fin_area_cm2,
        "footprint_area_cm2": footprint_cm2,
        "effective_extra_area_cm2": effective_extra_cm2,
        "average_fin_efficiency_percent": (100.0 * sum(effs) / len(effs)) if effs else 0.0,
        "thermal_segment_count": len(heatsink_fin_segment_specs(cfg)),
        "fins": specs,
    }


def heatsink_effective_extra_area_cm2(cfg: "PlateConfig") -> float:
    if not getattr(cfg, "heatsink_enabled", False):
        return 0.0
    if getattr(cfg, "heatsink_geometry_enabled", False):
        return float(heatsink_geometry_summary(cfg)["effective_extra_area_cm2"])
    extra = max(0.0, getattr(cfg, "heatsink_extra_area_cm2", 0.0))
    eff = max(0.0, min(100.0, getattr(cfg, "heatsink_efficiency_percent", 70.0))) / 100.0
    hmul = max(0.0, getattr(cfg, "heatsink_h_multiplier", 1.0))
    return extra * eff * hmul


def effective_convection_h_for_solver(cfg: "PlateConfig") -> float:
    """Return scalar h equivalent used for summaries/optimizer scoring.

    The actual solver uses a spatial cooling-loss map when geometry fins are
    enabled, so fin placement matters.
    """
    base_h = max(0.05, float(cfg.convection_h_w_m2k))
    if not getattr(cfg, "heatsink_enabled", False):
        return base_h

    plate_l_m = max(1e-6, cfg.plate_length_cm / 100.0)
    plate_w_m = max(1e-6, cfg.plate_width_cm / 100.0)
    base_area_m2 = max(1e-9, 2.0 * plate_l_m * plate_w_m)
    extra_area_m2 = heatsink_effective_extra_area_cm2(cfg) / 10000.0
    effective_area_m2 = base_area_m2 + extra_area_m2
    return base_h * effective_area_m2 / base_area_m2


def cooling_loss_coeff_map(cfg: "PlateConfig", shape: Tuple[int, int], dx_m: float, dy_m: float) -> np.ndarray:
    """Return loss coefficient map in W/m²K for each plate cell.

    For a plain plate, loss_coeff = 2*h because both large faces cool.
    For geometry fins, fin cooling is applied locally under each fin footprint.
    """
    base_h = max(0.05, float(cfg.convection_h_w_m2k))

    if getattr(cfg, "heatsink_enabled", False) and not getattr(cfg, "heatsink_geometry_enabled", False):
        return np.full(shape, 2.0 * effective_convection_h_for_solver(cfg), dtype=float)

    loss = np.full(shape, 2.0 * base_h, dtype=float)

    if not (getattr(cfg, "heatsink_enabled", False) and getattr(cfg, "heatsink_geometry_enabled", False)):
        return loss

    nx, ny = shape
    plate_l_m = cfg.plate_length_cm / 100.0
    plate_w_m = cfg.plate_width_cm / 100.0
    x = (np.arange(nx) + 0.5) * dx_m - plate_l_m / 2.0
    y = (np.arange(ny) + 0.5) * dy_m - plate_w_m / 2.0
    X, Y = np.meshgrid(x, y, indexing="ij")
    cell_area = dx_m * dy_m

    # v15 performance:
    # Apply fin cooling using index slices rather than building a full boolean
    # mask for every fin segment. This is much faster when many fins/segments
    # are present and gives the same local-cooling behavior.
    for spec in heatsink_fin_segment_specs(cfg, dx_m=dx_m, dy_m=dy_m):
        cx = spec["center_x_cm"] / 100.0
        cy = spec["center_y_cm"] / 100.0
        hx = (spec["footprint_x_cm"] / 100.0) / 2.0
        hy = (spec["footprint_y_cm"] / 100.0) / 2.0

        ix0 = int(np.searchsorted(x, cx - hx, side="left"))
        ix1 = int(np.searchsorted(x, cx + hx, side="right"))
        iy0 = int(np.searchsorted(y, cy - hy, side="left"))
        iy1 = int(np.searchsorted(y, cy + hy, side="right"))

        ix0 = max(0, min(len(x) - 1, ix0))
        ix1 = max(ix0 + 1, min(len(x), ix1))
        iy0 = max(0, min(len(y) - 1, iy0))
        iy1 = max(iy0 + 1, min(len(y), iy1))

        covered_area = float((ix1 - ix0) * (iy1 - iy0) * cell_area)
        eta = float(spec.get("full_fin_efficiency", fin_efficiency_for_spec(cfg, spec)))
        fin_conductance_w_k = base_h * eta * spec["raw_area_m2"]

        # Base map already counted the back face under the fin. That area is
        # replaced by fin conductance.
        add_coeff = fin_conductance_w_k / max(1e-12, covered_area) - base_h
        loss[ix0:ix1, iy0:iy1] += add_coeff

    return np.maximum(loss, 0.0)



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
    alpha = k / (rho * cp)
    x_tmp, y_tmp, _, _ = build_grid(cfg)
    loss = cooling_loss_coeff_map(cfg, (len(x_tmp), len(y_tmp)), dx_m, dy_m)
    beta_max = float(np.max(loss)) / (rho * cp * t)

    denom = 2.0 * alpha * ((1.0 / (dx_m * dx_m)) + (1.0 / (dy_m * dy_m))) + beta_max
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
    loss = cooling_loss_coeff_map(cfg, q_w_m2.shape, dx_m, dy_m)

    theta = np.zeros_like(q_w_m2, dtype=float)

    ax = k * t / (dx_m * dx_m)
    ay = k * t / (dy_m * dy_m)
    center = 2.0 * ax + 2.0 * ay + loss

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
    loss = cooling_loss_coeff_map(cfg, q_w_m2.shape, dx_m, dy_m)

    alpha = k / (rho * cp)
    beta = loss / (rho * cp * t_m)
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

    effective_h = effective_convection_h_for_solver(cfg)
    loss_map = cooling_loss_coeff_map(cfg, q_w_m2.shape, dx_m, dy_m)
    thermal_conductance_w_k = float(np.sum(loss_map * dx_m * dy_m))
    simple_average_final_rise_c = total_power_w / max(1e-12, thermal_conductance_w_k)

    tau_s = heat_capacity_j_c / max(1e-12, thermal_conductance_w_k)
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
            "convective_loss_w": float(np.sum(loss_map * theta * cell_area)),
        }

        for r, mask, covered_area_m2 in resistor_masks:
            est = resistor_temperature_estimate(cfg, r, temp, mask, covered_area_m2)
            row[f"{r.name}_avg_temp_c"] = est["plate_footprint_avg_temp_c"]
            row[f"{r.name}_max_temp_c"] = est["plate_footprint_max_temp_c"]
            row[f"{r.name}_case_max_temp_c"] = est["estimated_case_max_temp_c"]
            row[f"{r.name}_element_max_temp_c"] = est["estimated_element_max_temp_c"]

        snap_rows.append(row)

    resistor_reports = []
    final_snap = snapshots[-1]
    final_temp = final_snap.temp_c
    for r, mask, covered_area_m2 in resistor_masks:
        resistor_reports.append(resistor_temperature_estimate(cfg, r, final_temp, mask, covered_area_m2))

    hot_idx = np.unravel_index(np.argmax(final_temp), final_temp.shape)
    hottest_plate_point = {
        "x_cm": float(x_m[hot_idx[0]] * 100.0),
        "y_cm": float(y_m[hot_idx[1]] * 100.0),
        "temp_c": float(final_temp[hot_idx]),
    }

    precision_notes = []
    min_res_dim_mm = min([min(r.length_mm, r.width_mm) for r in cfg.resistors], default=999.0)
    if cfg.grid_mm > max(0.1, min_res_dim_mm / 2.0):
        precision_notes.append(
            "Grid size is large compared with the smallest resistor footprint. "
            "Use a smaller grid for better hotspot/contact resolution."
        )
    if getattr(cfg, "heatsink_geometry_enabled", False) and getattr(cfg, "heatsink_fin_count", 0) > 0:
        seg_count = heatsink_geometry_summary(cfg).get("thermal_segment_count", 0)
        if seg_count < max(1, getattr(cfg, "heatsink_fin_count", 0)):
            precision_notes.append("Fin thermal segmentation is very low. Use auto or increase thermal segments.")

    return {
        "total_power_w": total_power_w,
        "effective_h_w_m2k": cfg.convection_h_w_m2k,
        "advanced_cooling_enabled": cfg.advanced_cooling_enabled,
        "cooling_notes": cfg.cooling_notes,
        "plate_area_both_faces_cm2": top_bottom_area_m2 * 10000.0,
        "plate_area_with_edges_cm2": (top_bottom_area_m2 + edge_area_m2) * 10000.0,
        "mass_kg": mass_kg,
        "heat_capacity_j_per_c": heat_capacity_j_c,
        "base_h_w_m2k": cfg.convection_h_w_m2k,
        "effective_convection_h_w_m2k": effective_h,
        "heatsink_enabled": getattr(cfg, "heatsink_enabled", False),
        "heatsink_raw_extra_area_cm2": getattr(cfg, "heatsink_extra_area_cm2", 0.0),
        "heatsink_effective_extra_area_cm2": heatsink_effective_extra_area_cm2(cfg),
        "heatsink_geometry": heatsink_geometry_summary(cfg),
        "resistor_case_to_plate_cw": float(getattr(cfg, "resistor_case_to_plate_cw", 0.5)),
        "resistor_element_to_case_cw": float(getattr(cfg, "resistor_element_to_case_cw", 0.6)),
        "hottest_plate_point": hottest_plate_point,
        "precision_notes": precision_notes,
        "thermal_conductance_w_per_k": thermal_conductance_w_k,
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




FIELD_HELP_TEXT = """Field meanings

Plate width/X cm:
  Horizontal size of the metal plate shown on screen. Bigger width gives more cooling area and more spacing room.

Plate height/Y cm:
  Vertical size of the metal plate shown on screen. Bigger height gives more cooling area and more spacing room.

Thickness mm:
  Metal thickness. More thickness improves heat spreading and thermal mass. It does not replace exposed surface area for continuous cooling.

Material:
  Template material. Template values lock k, density, and heat capacity. Choose custom if you want to edit them.

k W/mK:
  Thermal conductivity. Higher k spreads heat better across the plate. Aluminium is about 205, copper about 385, steel about 45.

Density kg/m³:
  Used for mass and warm-up behavior. Higher density usually means more thermal mass for the same volume.

Heat cap J/kgK:
  Specific heat capacity. Higher means the plate warms up more slowly.

Ambient °C:
  Air temperature around the plate. All predicted temperatures rise directly with this.

h W/m²K:
  Effective passive cooling strength to air. Lower h is worse cooling. For passive builds, h=5 is pessimistic, h=7 normal-ish, h=10 optimistic/good open air.

Grid mm:
  Simulation cell size. Smaller is more detailed but slower. 5 mm is a good start. 2.5 mm is detailed. 10 mm is rough but fast.

Initial °C:
  Starting plate temperature for the time-based warm-up simulation.

Max time:
  How long the transient simulation runs, e.g. 30m or 1h.

Snapshot every:
  Time distance between saved slider steps, e.g. 30s, 1m, 5m.

Resistor power W:
  Heat produced by that resistor. For safety testing, use worst-case power.

Resistor x/y cm:
  Center position of the resistor. x is horizontal, y is vertical. The plate center is 0,0.

Resistor length/width mm:
  Contact footprint touching the plate. This is the heat injection area, not necessarily the whole resistor body.

Advanced cooling:
  Turns real-world placement into an effective h estimate. It is still an estimate; real temperature testing is required.

Deep optimize:
  v6 uses a deeper two-stage optimizer. First it generates many candidate layouts and scores them with a fast thermal-interaction model. Then it runs real coarse steady-state heat solves on the best candidates and locally refines the best layout. Use Deep for normal work; Extreme searches harder but can take much longer.

Optimizer grid mm:
  Cell size used only while optimizing locations. Larger is faster but rougher. 12-15 mm is good for deep search; 8-10 mm is more accurate but slower. The final Run simulation still uses the main Grid mm field.
"""


class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(self._tip, text=self.text, justify="left", relief="solid", borderwidth=1, padding=6, wraplength=360)
            label.pack()
        except Exception:
            self._tip = None

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def _center_bounds_for_resistor(
    r: Resistor,
    plate_x_cm: float,
    plate_y_cm: float,
    margin_cm: float,
) -> Tuple[float, float, float, float]:
    """Allowed center coordinate range for one resistor."""
    half_l_cm = r.length_mm / 20.0
    half_w_cm = r.width_mm / 20.0

    xmin = -plate_x_cm / 2.0 + half_l_cm + margin_cm
    xmax =  plate_x_cm / 2.0 - half_l_cm - margin_cm
    ymin = -plate_y_cm / 2.0 + half_w_cm + margin_cm
    ymax =  plate_y_cm / 2.0 - half_w_cm - margin_cm

    if xmin > xmax or ymin > ymax:
        raise ValueError(
            f"{r.name} does not fit on the plate with the current margin. "
            "Reduce margin, reduce resistor size, or use a larger plate."
        )

    return xmin, xmax, ymin, ymax


def _rectangles_overlap(
    r1: Resistor,
    p1: Tuple[float, float],
    r2: Resistor,
    p2: Tuple[float, float],
    clearance_cm: float = 0.15,
) -> bool:
    """Axis-aligned footprint overlap check."""
    x1, y1 = p1
    x2, y2 = p2
    half_l_1 = r1.length_mm / 20.0
    half_w_1 = r1.width_mm / 20.0
    half_l_2 = r2.length_mm / 20.0
    half_w_2 = r2.width_mm / 20.0

    return (
        abs(x1 - x2) < (half_l_1 + half_l_2 + clearance_cm)
        and abs(y1 - y2) < (half_w_1 + half_w_2 + clearance_cm)
    )


def _layout_is_valid(
    resistors: List[Resistor],
    positions: List[Tuple[float, float]],
    plate_x_cm: float,
    plate_y_cm: float,
    margin_cm: float,
    clearance_cm: float = 0.15,
) -> bool:
    if len(resistors) != len(positions):
        return False

    for r, (x, y) in zip(resistors, positions):
        xmin, xmax, ymin, ymax = _center_bounds_for_resistor(r, plate_x_cm, plate_y_cm, margin_cm)
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

    for i in range(len(resistors)):
        for j in range(i + 1, len(resistors)):
            if _rectangles_overlap(resistors[i], positions[i], resistors[j], positions[j], clearance_cm):
                return False

    return True


def _thermal_interaction_objective(
    resistors: List[Resistor],
    positions: List[Tuple[float, float]],
    plate_x_cm: float,
    plate_y_cm: float,
    k_w_mk: float,
    thickness_mm: float,
    h_w_m2k: float,
    margin_cm: float,
) -> float:
    """
    Fast approximate layout score.

    Lower is better.

    This is not the full finite-difference solver. It is a fast optimizer score
    designed to find sensible source placement before the real simulation.

    It considers:
      - heat sources heating each other
      - edge/boundary penalty from reduced spreading room
      - layout coverage across both X and Y
      - power weighting
    """
    if not _layout_is_valid(resistors, positions, plate_x_cm, plate_y_cm, margin_cm):
        return float("inf")

    total_power = max(1e-9, sum(max(0.0, r.power_w) for r in resistors))

    # Thin plate with convection has a useful thermal spreading length:
    # L = sqrt(k*t/(2h)).
    # In plain words: how far heat tends to spread before air cooling dominates.
    t_m = max(0.0001, thickness_mm / 1000.0)
    h = max(0.05, h_w_m2k)
    spread_len_cm = math.sqrt(max(1e-12, k_w_mk * t_m / (2.0 * h))) * 100.0
    spread_len_cm = max(2.0, min(spread_len_cm, max(plate_x_cm, plate_y_cm) * 2.0))

    objective = 0.0

    # Pair interaction: hot sources close together are bad.
    for i, (ri, pi) in enumerate(zip(resistors, positions)):
        xi, yi = pi
        ri_eff_cm = max(0.2, math.sqrt((ri.length_mm / 10.0) * (ri.width_mm / 10.0) / math.pi))

        local = ri.power_w / max(0.2, ri_eff_cm)

        for j, (rj, pj) in enumerate(zip(resistors, positions)):
            if i == j:
                continue
            xj, yj = pj
            d = max(0.05, math.hypot(xi - xj, yi - yj))
            # Exponential decay gives a stable approximation of "thermal influence".
            local += rj.power_w * math.exp(-d / spread_len_cm) / math.sqrt(d + ri_eff_cm)

        # Edge penalty. A source near an insulated/thin edge has less metal to
        # spread into. Approximate that with mirrored heat sources across edges.
        left = xi + plate_x_cm / 2.0 - ri.length_mm / 20.0
        right = plate_x_cm / 2.0 - xi - ri.length_mm / 20.0
        bottom = yi + plate_y_cm / 2.0 - ri.width_mm / 20.0
        top = plate_y_cm / 2.0 - yi - ri.width_mm / 20.0

        for edge_dist in (left, right, bottom, top):
            edge_dist = max(0.05, edge_dist)
            mirror_d = 2.0 * edge_dist
            local += 0.45 * ri.power_w * math.exp(-mirror_d / spread_len_cm) / math.sqrt(mirror_d + ri_eff_cm)

        # Optimize for the hottest source, not just the average.
        objective = max(objective, local)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    usable_x = max(0.1, plate_x_cm - 2.0 * margin_cm)
    usable_y = max(0.1, plate_y_cm - 2.0 * margin_cm)
    span_x = (max(xs) - min(xs)) / usable_x if len(xs) > 1 else 0.0
    span_y = (max(ys) - min(ys)) / usable_y if len(ys) > 1 else 0.0

    # Coverage penalty prevents the bad "all in one line" solution when the
    # plate has room in both dimensions. This is the key fix compared with v5.
    if len(resistors) >= 3 and usable_x > 8.0 and usable_y > 8.0:
        desired_x = 0.65
        desired_y = 0.65
        coverage_penalty = total_power * (
            max(0.0, desired_x - span_x) ** 2
            + max(0.0, desired_y - span_y) ** 2
        )
        objective += 0.35 * coverage_penalty

    # Keep the center of heat roughly near the center of the plate, unless
    # asymmetrical resistor powers naturally pull it elsewhere.
    heat_cx = sum(r.power_w * p[0] for r, p in zip(resistors, positions)) / total_power
    heat_cy = sum(r.power_w * p[1] for r, p in zip(resistors, positions)) / total_power
    objective += 0.05 * total_power * ((heat_cx / usable_x) ** 2 + (heat_cy / usable_y) ** 2)

    return objective


def _make_grid_layout(
    resistors: List[Resistor],
    plate_x_cm: float,
    plate_y_cm: float,
    margin_cm: float,
    rows: int,
    cols: int,
) -> Optional[List[Tuple[float, float]]]:
    """Create a rows×cols layout and choose the best cells if there are extras."""
    count = len(resistors)
    if rows < 1 or cols < 1 or rows * cols < count:
        return None

    max_l_cm = max(r.length_mm for r in resistors) / 10.0
    max_w_cm = max(r.width_mm for r in resistors) / 10.0

    xmin = -plate_x_cm / 2.0 + max_l_cm / 2.0 + margin_cm
    xmax =  plate_x_cm / 2.0 - max_l_cm / 2.0 - margin_cm
    ymin = -plate_y_cm / 2.0 + max_w_cm / 2.0 + margin_cm
    ymax =  plate_y_cm / 2.0 - max_w_cm / 2.0 - margin_cm

    if xmin > xmax or ymin > ymax:
        return None

    xs = [0.0] if cols == 1 else list(np.linspace(xmin, xmax, cols))
    ys = [0.0] if rows == 1 else list(np.linspace(ymin, ymax, rows))
    cells = [(x, y) for y in ys for x in xs]

    # Prefer using outer cells first when there are extra cells. This gives
    # better spreading than simply taking the first N cells.
    cells = sorted(cells, key=lambda p: math.hypot(p[0], p[1]), reverse=True)

    if len(cells) <= 12 and len(cells) > count:
        best = None
        best_score = -1e99
        for combo in itertools.combinations(cells, count):
            combo = list(combo)
            xs2 = [p[0] for p in combo]
            ys2 = [p[1] for p in combo]
            span = (max(xs2) - min(xs2)) + (max(ys2) - min(ys2))
            balance = -abs((max(xs2) + min(xs2))) - abs((max(ys2) + min(ys2)))
            min_pair = min(
                math.hypot(combo[i][0] - combo[j][0], combo[i][1] - combo[j][1])
                for i in range(count)
                for j in range(i + 1, count)
            ) if count > 1 else 0.0
            score = 2.0 * min_pair + 0.3 * span + 0.2 * balance
            if score > best_score:
                best_score = score
                best = combo
        cells = best if best is not None else cells[:count]
    else:
        cells = cells[:count]

    # Assign highest power resistors to the cells with most spreading room.
    # For equal power, this simply preserves a stable ordering.
    order = sorted(range(count), key=lambda i: resistors[i].power_w, reverse=True)
    cells_sorted = sorted(cells, key=lambda p: min(
        p[0] + plate_x_cm / 2.0,
        plate_x_cm / 2.0 - p[0],
        p[1] + plate_y_cm / 2.0,
        plate_y_cm / 2.0 - p[1],
    ), reverse=True)

    assigned = [None] * count
    for res_i, pos in zip(order, cells_sorted):
        assigned[res_i] = pos

    if not _layout_is_valid(resistors, assigned, plate_x_cm, plate_y_cm, margin_cm):
        return None

    return assigned


def evenly_spaced_positions(
    count: int,
    plate_x_cm: float,
    plate_y_cm: float,
    footprint_l_mm: float,
    footprint_w_mm: float,
    margin_cm: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Return an even 2D layout.

    v5.2 fix:
    The previous version often preferred a single line because it maximized
    nearest-neighbor distance. This version also considers plate aspect ratio
    and uses both dimensions when the plate has room.
    """
    if count <= 0:
        return []

    dummy = [
        Resistor(f"R{i + 1}", 1.0, 0.0, 0.0, footprint_l_mm, footprint_w_mm)
        for i in range(count)
    ]

    best = None
    best_score = -1e99
    target_aspect = max(0.01, plate_x_cm / max(0.01, plate_y_cm))

    for rows in range(1, count + 1):
        for cols in range(1, count + 1):
            if rows * cols < count:
                continue

            layout = _make_grid_layout(dummy, plate_x_cm, plate_y_cm, margin_cm, rows, cols)
            if layout is None:
                continue

            grid_aspect = cols / rows
            unused = rows * cols - count
            xs = [p[0] for p in layout]
            ys = [p[1] for p in layout]
            span_x = max(xs) - min(xs) if len(xs) > 1 else 0.0
            span_y = max(ys) - min(ys) if len(ys) > 1 else 0.0
            min_pair = min(
                math.hypot(layout[i][0] - layout[j][0], layout[i][1] - layout[j][1])
                for i in range(count)
                for j in range(i + 1, count)
            ) if count > 1 else 0.0

            # Prefer grid aspect matching plate aspect, low unused cells, and
            # actual 2D coverage.
            aspect_error = abs(math.log(max(0.01, grid_aspect / target_aspect)))
            coverage = span_x + span_y
            score = 1.2 * min_pair + 0.25 * coverage - 2.0 * aspect_error - 0.5 * unused

            if score > best_score:
                best_score = score
                best = layout

    if best is None:
        raise ValueError("Could not create an even layout.")

    return best


def candidate_layouts_for_count(
    count: int,
    plate_x_cm: float,
    plate_y_cm: float,
    footprint_l_mm: float,
    footprint_w_mm: float,
    margin_cm: float,
) -> List[List[Tuple[float, float]]]:
    """
    Generate sensible candidate layouts.

    Kept for compatibility, but improved in v5.2.
    """
    dummy = [
        Resistor(f"R{i + 1}", 1.0, 0.0, 0.0, footprint_l_mm, footprint_w_mm)
        for i in range(count)
    ]
    return generate_candidate_layouts(dummy, plate_x_cm, plate_y_cm, margin_cm)


def generate_candidate_layouts(
    resistors: List[Resistor],
    plate_x_cm: float,
    plate_y_cm: float,
    margin_cm: float,
) -> List[List[Tuple[float, float]]]:
    """Generate many deterministic starting layouts for the optimizer."""
    count = len(resistors)
    layouts: List[List[Tuple[float, float]]] = []
    seen = set()

    def add(layout):
        if layout is None:
            return
        if not _layout_is_valid(resistors, layout, plate_x_cm, plate_y_cm, margin_cm):
            return
        key = tuple((round(x, 3), round(y, 3)) for x, y in layout)
        if key not in seen:
            seen.add(key)
            layouts.append(layout)

    # Current-ish geometrical grids.
    for rows in range(1, count + 1):
        for cols in range(1, count + 1):
            if rows * cols >= count:
                add(_make_grid_layout(resistors, plate_x_cm, plate_y_cm, margin_cm, rows, cols))

    # Try different margins. Sometimes being slightly less close to the edge
    # wins thermally even if the official margin is small.
    max_l = max(r.length_mm for r in resistors)
    max_w = max(r.width_mm for r in resistors)
    for m in [margin_cm, margin_cm + 1.0, margin_cm + 2.0, max(0.0, margin_cm / 2.0), 0.0]:
        try:
            pos = evenly_spaced_positions(count, plate_x_cm, plate_y_cm, max_l, max_w, m)
            if _layout_is_valid(resistors, pos, plate_x_cm, plate_y_cm, margin_cm):
                add(pos)
        except Exception:
            pass

    # Ring / ellipse layout, useful for 3+ sources.
    if count >= 3:
        max_l_cm = max(r.length_mm for r in resistors) / 10.0
        max_w_cm = max(r.width_mm for r in resistors) / 10.0
        rx = max(0.1, plate_x_cm / 2.0 - max_l_cm / 2.0 - margin_cm)
        ry = max(0.1, plate_y_cm / 2.0 - max_w_cm / 2.0 - margin_cm)

        for scale in [0.55, 0.70, 0.85]:
            layout = []
            for i in range(count):
                angle = 2.0 * math.pi * i / count
                layout.append((scale * rx * math.cos(angle), scale * ry * math.sin(angle)))
            add(layout)

            # Rotated version.
            layout = []
            for i in range(count):
                angle = 2.0 * math.pi * (i + 0.5) / count
                layout.append((scale * rx * math.cos(angle), scale * ry * math.sin(angle)))
            add(layout)

    return layouts


def optimize_layout_fast(
    resistors: List[Resistor],
    plate_x_cm: float,
    plate_y_cm: float,
    k_w_mk: float,
    thickness_mm: float,
    h_w_m2k: float,
    margin_cm: float,
    progress_callback=None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[List[Tuple[float, float]], float, int]:
    """
    Improved v5.2 optimizer.

    Uses:
      1. deterministic grid/ellipse candidates
      2. random valid starts
      3. simulated-annealing style local movement

    The objective is a fast thermal-interaction estimate. The full simulation
    should still be run afterward for the final result.
    """
    count = len(resistors)
    if count == 0:
        raise ValueError("No resistors to optimize.")

    rng = np.random.default_rng(12345)

    candidates = generate_candidate_layouts(resistors, plate_x_cm, plate_y_cm, margin_cm)

    bounds = [_center_bounds_for_resistor(r, plate_x_cm, plate_y_cm, margin_cm) for r in resistors]

    def random_layout(max_attempts=4000):
        positions = []
        order = sorted(range(count), key=lambda i: resistors[i].power_w, reverse=True)
        assigned = [None] * count

        for idx in order:
            xmin, xmax, ymin, ymax = bounds[idx]
            placed = False
            for _ in range(max_attempts):
                p = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
                ok = True
                for j in range(count):
                    if assigned[j] is None:
                        continue
                    if _rectangles_overlap(resistors[idx], p, resistors[j], assigned[j], 0.15):
                        ok = False
                        break
                if ok:
                    assigned[idx] = p
                    placed = True
                    break
            if not placed:
                return None

        return assigned

    # Add random starts.
    random_start_count = min(250, max(50, 25 * count))
    for _ in range(random_start_count):
        layout = random_layout()
        if layout is not None and _layout_is_valid(resistors, layout, plate_x_cm, plate_y_cm, margin_cm):
            candidates.append(layout)

    best_layout = None
    best_score = float("inf")
    tried = 0

    def score(layout):
        return _thermal_interaction_objective(
            resistors=resistors,
            positions=layout,
            plate_x_cm=plate_x_cm,
            plate_y_cm=plate_y_cm,
            k_w_mk=k_w_mk,
            thickness_mm=thickness_mm,
            h_w_m2k=h_w_m2k,
            margin_cm=margin_cm,
        )

    # Evaluate starts.
    unique = []
    seen = set()
    for layout in candidates:
        key = tuple((round(x, 2), round(y, 2)) for x, y in layout)
        if key in seen:
            continue
        seen.add(key)
        unique.append(layout)

    for layout in unique:
        tried += 1
        s = score(layout)
        if s < best_score:
            best_score = s
            best_layout = [tuple(p) for p in layout]

    if best_layout is None:
        raise ValueError("Could not find any valid starting layout.")

    # Local improvement from the best several starts.
    starts = sorted(unique, key=score)[:min(12, len(unique))]
    max_span = max(plate_x_cm, plate_y_cm)
    moves_per_start = max(600, 250 * count)

    for start_i, start in enumerate(starts, 1):
        current = [tuple(p) for p in start]
        current_score = score(current)
        step_cm = max(1.0, max_span * 0.18)
        temperature = max(0.1, current_score * 0.08)

        for move in range(moves_per_start):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Optimization was cancelled.")

            i = int(rng.integers(0, count))
            xmin, xmax, ymin, ymax = bounds[i]
            old_p = current[i]

            # Mix small nudges and total repositions.
            if rng.random() < 0.15:
                new_p = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
            else:
                new_p = (
                    float(min(xmax, max(xmin, old_p[0] + rng.normal(0.0, step_cm)))),
                    float(min(ymax, max(ymin, old_p[1] + rng.normal(0.0, step_cm)))),
                )

            trial = list(current)
            trial[i] = new_p

            if not _layout_is_valid(resistors, trial, plate_x_cm, plate_y_cm, margin_cm):
                continue

            trial_score = score(trial)
            tried += 1

            accept = trial_score < current_score
            if not accept:
                # Annealing escape from local minima.
                delta = trial_score - current_score
                if rng.random() < math.exp(-delta / max(1e-9, temperature)):
                    accept = True

            if accept:
                current = trial
                current_score = trial_score

                if current_score < best_score:
                    best_score = current_score
                    best_layout = [tuple(p) for p in current]

            step_cm *= 0.996
            temperature *= 0.997

        if progress_callback is not None:
            progress_callback(
                f"Optimizer refined start {start_i}/{len(starts)}. "
                f"Best score: {best_score:.3f}. Tried {tried} layouts/moves."
            )

    return best_layout, best_score, tried


def _optimizer_cfg_with_positions(cfg: PlateConfig, positions: List[Tuple[float, float]], grid_mm: float) -> PlateConfig:
    """Create a temporary config used only by the optimizer's coarse heat solve."""
    rs = [
        Resistor(
            name=r.name,
            power_w=r.power_w,
            center_x_cm=float(p[0]),
            center_y_cm=float(p[1]),
            length_mm=r.length_mm,
            width_mm=r.width_mm,
        )
        for r, p in zip(cfg.resistors, positions)
    ]

    return PlateConfig(
        plate_length_cm=cfg.plate_length_cm,
        plate_width_cm=cfg.plate_width_cm,
        plate_thickness_mm=cfg.plate_thickness_mm,
        material_name=cfg.material_name,
        thermal_conductivity_w_mk=cfg.thermal_conductivity_w_mk,
        density_kg_m3=cfg.density_kg_m3,
        heat_capacity_j_kgk=cfg.heat_capacity_j_kgk,
        ambient_c=cfg.ambient_c,
        convection_h_w_m2k=effective_convection_h_for_solver(cfg),
        grid_mm=grid_mm,
        resistors=rs,
        initial_plate_temp_c=cfg.initial_plate_temp_c,
        max_time_s=cfg.max_time_s,
        snapshot_every_s=cfg.snapshot_every_s,
        include_steady_state=False,
    )


def _layout_key(positions: List[Tuple[float, float]], decimals: int = 2) -> Tuple[Tuple[float, float], ...]:
    return tuple((round(float(x), decimals), round(float(y), decimals)) for x, y in positions)


def _score_layout_by_coarse_heat_solve(
    cfg: PlateConfig,
    positions: List[Tuple[float, float]],
    margin_cm: float,
    grid_mm: float,
    cache: Dict,
    stop_event: Optional[threading.Event] = None,
    max_iter: int = 1800,
    tolerance_c: float = 0.035,
) -> Tuple[float, Dict]:
    """
    Score a layout by running an actual coarse steady-state plate heat solve.

    This is slower than the analytical thermal-interaction objective, but it is
    much closer to the real simulator behavior. It is used only by the optimizer,
    not by the normal final simulation.
    """
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Optimization was cancelled.")

    if not _layout_is_valid(cfg.resistors, positions, cfg.plate_length_cm, cfg.plate_width_cm, margin_cm):
        return float("inf"), {"reason": "invalid"}

    key = (round(grid_mm, 3), _layout_key(positions, 2))
    if key in cache:
        return cache[key]

    opt_cfg = _optimizer_cfg_with_positions(cfg, positions, grid_mm)
    x, y, dx, dy = build_grid(opt_cfg)
    q, masks = add_heat_sources(opt_cfg, x, y, dx, dy)

    temp, info = solve_steady_state(
        opt_cfg,
        q,
        dx,
        dy,
        progress_callback=None,
        stop_event=stop_event,
        max_iter=max_iter,
        tolerance_c=tolerance_c,
    )

    plate_max = float(np.max(temp))
    plate_avg = float(np.mean(temp))
    resistor_maxes = []
    resistor_avgs = []

    for r, mask, covered_area in masks:
        resistor_maxes.append(float(np.max(temp[mask])))
        resistor_avgs.append(float(np.mean(temp[mask])))

    hottest_res = max(resistor_maxes) if resistor_maxes else plate_max

    # Main goal: reduce the worst hotspot.
    # Small tie-breakers:
    #   - lower average plate temp is slightly better
    #   - lower spread between resistor temps is slightly better
    # This avoids ugly layouts that have one resistor clearly suffering.
    score = max(plate_max, hottest_res)
    score += 0.015 * plate_avg
    if len(resistor_maxes) > 1:
        score += 0.04 * float(np.std(resistor_maxes))

    result = (
        score,
        {
            "plate_max_c": plate_max,
            "plate_avg_c": plate_avg,
            "hottest_resistor_c": hottest_res,
            "resistor_maxes_c": resistor_maxes,
            "iterations": info.get("iterations"),
            "converged": info.get("converged"),
            "grid_cells": int(len(x) * len(y)),
        },
    )
    cache[key] = result
    return result


def _random_valid_layout(
    resistors: List[Resistor],
    plate_x_cm: float,
    plate_y_cm: float,
    margin_cm: float,
    rng,
    max_attempts_per_resistor: int = 2500,
) -> Optional[List[Tuple[float, float]]]:
    bounds = [_center_bounds_for_resistor(r, plate_x_cm, plate_y_cm, margin_cm) for r in resistors]
    order = sorted(range(len(resistors)), key=lambda i: resistors[i].power_w, reverse=True)
    assigned: List[Optional[Tuple[float, float]]] = [None] * len(resistors)

    for idx in order:
        xmin, xmax, ymin, ymax = bounds[idx]
        placed = False
        for _ in range(max_attempts_per_resistor):
            p = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
            ok = True
            for j, other in enumerate(assigned):
                if other is None:
                    continue
                if _rectangles_overlap(resistors[idx], p, resistors[j], other, 0.15):
                    ok = False
                    break
            if ok:
                assigned[idx] = p
                placed = True
                break
        if not placed:
            return None

    return [(float(x), float(y)) for x, y in assigned]  # type: ignore[arg-type]


def _jitter_layout(
    layout: List[Tuple[float, float]],
    resistors: List[Resistor],
    plate_x_cm: float,
    plate_y_cm: float,
    margin_cm: float,
    rng,
    sigma_cm: float,
    attempts: int = 40,
) -> Optional[List[Tuple[float, float]]]:
    bounds = [_center_bounds_for_resistor(r, plate_x_cm, plate_y_cm, margin_cm) for r in resistors]

    for _ in range(attempts):
        out = []
        for (x, y), (xmin, xmax, ymin, ymax) in zip(layout, bounds):
            nx = min(xmax, max(xmin, float(x + rng.normal(0.0, sigma_cm))))
            ny = min(ymax, max(ymin, float(y + rng.normal(0.0, sigma_cm))))
            out.append((nx, ny))
        if _layout_is_valid(resistors, out, plate_x_cm, plate_y_cm, margin_cm):
            return out

    return None


def _locally_refine_layout_with_heat_solve(
    cfg: PlateConfig,
    start_layout: List[Tuple[float, float]],
    margin_cm: float,
    grid_mm: float,
    cache: Dict,
    step_sizes_cm: List[float],
    stop_event: Optional[threading.Event],
    progress_callback=None,
    label: str = "coarse",
    max_passes_per_step: int = 2,
) -> Tuple[List[Tuple[float, float]], float, Dict, int]:
    """
    Coordinate/pattern-search refinement using actual coarse heat-solve scoring.

    It tries moving each resistor in 8 directions, keeps the best improvement,
    then repeats with smaller steps. This is slower than the fast optimizer but
    much less likely to get stuck in obviously bad layouts.
    """
    current = [tuple(p) for p in start_layout]
    current_score, current_info = _score_layout_by_coarse_heat_solve(
        cfg, current, margin_cm, grid_mm, cache, stop_event=stop_event
    )
    evaluations = 1

    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    ]

    bounds = [_center_bounds_for_resistor(r, cfg.plate_length_cm, cfg.plate_width_cm, margin_cm) for r in cfg.resistors]

    for step_i, step in enumerate(step_sizes_cm, 1):
        improved_this_step = True
        passes = 0

        while improved_this_step and passes < max_passes_per_step:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Optimization was cancelled.")

            passes += 1
            improved_this_step = False
            best_trial = current
            best_trial_score = current_score
            best_trial_info = current_info

            # Try hottest resistors first, if we know the current footprint temps.
            order = list(range(len(cfg.resistors)))
            maxes = current_info.get("resistor_maxes_c")
            if isinstance(maxes, list) and len(maxes) == len(order):
                order = sorted(order, key=lambda i: maxes[i], reverse=True)

            for i in order:
                xmin, xmax, ymin, ymax = bounds[i]

                for dx_dir, dy_dir in directions:
                    old_x, old_y = current[i]
                    # Diagonal moves are scaled so they are not longer than axis moves.
                    diag_scale = 0.7071 if dx_dir != 0.0 and dy_dir != 0.0 else 1.0
                    nx = min(xmax, max(xmin, old_x + dx_dir * step * diag_scale))
                    ny = min(ymax, max(ymin, old_y + dy_dir * step * diag_scale))

                    trial = list(current)
                    trial[i] = (nx, ny)

                    if not _layout_is_valid(cfg.resistors, trial, cfg.plate_length_cm, cfg.plate_width_cm, margin_cm):
                        continue

                    trial_score, trial_info = _score_layout_by_coarse_heat_solve(
                        cfg, trial, margin_cm, grid_mm, cache, stop_event=stop_event
                    )
                    evaluations += 1

                    if trial_score < best_trial_score - 0.002:
                        best_trial = trial
                        best_trial_score = trial_score
                        best_trial_info = trial_info

            if best_trial_score < current_score - 0.002:
                current = [tuple(p) for p in best_trial]
                current_score = best_trial_score
                current_info = best_trial_info
                improved_this_step = True

        if progress_callback is not None:
            progress_callback(
                f"{label} refine step {step_i}/{len(step_sizes_cm)} "
                f"step={step:.2f} cm, best hotspot≈{current_info.get('plate_max_c', current_score):.1f} °C"
            )

    return current, current_score, current_info, evaluations


def optimize_layout_deep(
    cfg: PlateConfig,
    margin_cm: float,
    optimizer_grid_mm: float = 12.0,
    depth: str = "Deep",
    progress_callback=None,
    stop_event: Optional[threading.Event] = None,
    worker_count: int = 1,
) -> Tuple[List[Tuple[float, float]], float, int, Dict]:
    """
    v6 deeper optimizer.

    Stages:
      1. Generate deterministic layouts: grids, rings, ellipses.
      2. Add many random valid layouts and jittered variants.
      3. Rank with the fast thermal-interaction objective.
      4. Run actual coarse finite-difference steady-state heat solves on the best candidates.
      5. Locally refine using actual coarse heat-solve scoring.
      6. Re-score the best layout on a finer optimizer grid.

    It still cannot prove a global optimum, but it is far deeper than v5.2 and
    should avoid shallow "looks spaced but not thermally good" answers.
    """
    depth_key = str(depth).strip().lower()
    if "extreme" in depth_key:
        settings = {
            "random_count": 4500,
            "jitter_count": 900,
            "top_fast": 220,
            "top_fd": 100,
            "starts_to_refine": 8,
            "step_factor": [0.22, 0.14, 0.09, 0.055, 0.032, 0.018, 0.010],
            "passes": 3,
            "fine_grid_factor": 0.70,
        }
    elif "balanced" in depth_key or "fast" in depth_key:
        settings = {
            "random_count": 900,
            "jitter_count": 220,
            "top_fast": 90,
            "top_fd": 35,
            "starts_to_refine": 3,
            "step_factor": [0.18, 0.10, 0.055, 0.030, 0.016],
            "passes": 2,
            "fine_grid_factor": 0.85,
        }
    else:
        settings = {
            "random_count": 2200,
            "jitter_count": 520,
            "top_fast": 150,
            "top_fd": 65,
            "starts_to_refine": 5,
            "step_factor": [0.20, 0.125, 0.075, 0.045, 0.025, 0.014],
            "passes": 2,
            "fine_grid_factor": 0.75,
        }

    count = len(cfg.resistors)
    if count < 1:
        raise ValueError("No resistors to optimize.")

    rng = np.random.default_rng(20260427)
    plate_x_cm = cfg.plate_length_cm
    plate_y_cm = cfg.plate_width_cm

    optimizer_grid_mm = max(3.0, float(optimizer_grid_mm))
    coarse_grid_mm = optimizer_grid_mm
    fine_grid_mm = max(3.0, optimizer_grid_mm * settings["fine_grid_factor"])
    try:
        auto_workers = max(1, min(8, os.cpu_count() or 1))
        worker_count = int(worker_count)
        if worker_count <= 0:
            worker_count = auto_workers
        worker_count = max(1, min(worker_count, auto_workers))
    except Exception:
        worker_count = 1

    if progress_callback is not None:
        progress_callback(
            f"Deep optimizer started: depth={depth}, coarse grid={coarse_grid_mm:g} mm, "
            f"fine check grid={fine_grid_mm:g} mm, workers={worker_count}"
        )

    candidates = generate_candidate_layouts(cfg.resistors, plate_x_cm, plate_y_cm, margin_cm)

    # Include current layout as a candidate.
    current_layout = [(r.center_x_cm, r.center_y_cm) for r in cfg.resistors]
    if _layout_is_valid(cfg.resistors, current_layout, plate_x_cm, plate_y_cm, margin_cm):
        candidates.append(current_layout)

    # Include result from v5.2 fast optimizer as one strong seed.
    try:
        fast_best, fast_score, fast_tried = optimize_layout_fast(
            resistors=cfg.resistors,
            plate_x_cm=plate_x_cm,
            plate_y_cm=plate_y_cm,
            k_w_mk=cfg.thermal_conductivity_w_mk,
            thickness_mm=cfg.plate_thickness_mm,
            h_w_m2k=effective_convection_h_for_solver(cfg),
            margin_cm=margin_cm,
            progress_callback=None,
            stop_event=stop_event,
        )
        candidates.append(fast_best)
    except Exception:
        pass

    # Add random valid starts.
    for i in range(settings["random_count"]):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Optimization was cancelled.")
        layout = _random_valid_layout(cfg.resistors, plate_x_cm, plate_y_cm, margin_cm, rng)
        if layout is not None:
            candidates.append(layout)

    # De-duplicate before jitter.
    unique = []
    seen = set()
    for layout in candidates:
        if not _layout_is_valid(cfg.resistors, layout, plate_x_cm, plate_y_cm, margin_cm):
            continue
        key = _layout_key(layout, 2)
        if key not in seen:
            seen.add(key)
            unique.append(layout)

    # Rank with fast objective and keep a useful top group.
    def fast_score(layout):
        return _thermal_interaction_objective(
            resistors=cfg.resistors,
            positions=layout,
            plate_x_cm=plate_x_cm,
            plate_y_cm=plate_y_cm,
            k_w_mk=cfg.thermal_conductivity_w_mk,
            thickness_mm=cfg.plate_thickness_mm,
            h_w_m2k=effective_convection_h_for_solver(cfg),
            margin_cm=margin_cm,
        )

    unique = sorted(unique, key=fast_score)

    # Jitter around good starts. This explores "nearby but not obvious" positions.
    max_span = max(plate_x_cm, plate_y_cm)
    jitter_sources = unique[:min(30, len(unique))]
    for _ in range(settings["jitter_count"]):
        if not jitter_sources:
            break
        base = jitter_sources[int(rng.integers(0, len(jitter_sources)))]
        sigma = float(rng.uniform(0.015, 0.12) * max_span)
        layout = _jitter_layout(
            base,
            cfg.resistors,
            plate_x_cm,
            plate_y_cm,
            margin_cm,
            rng,
            sigma_cm=sigma,
        )
        if layout is not None:
            key = _layout_key(layout, 2)
            if key not in seen:
                seen.add(key)
                unique.append(layout)

    unique = sorted(unique, key=fast_score)
    fast_shortlist = unique[:min(settings["top_fast"], len(unique))]

    if progress_callback is not None:
        progress_callback(
            f"Generated {len(unique)} valid layouts. "
            f"Heat-solving the best {min(settings['top_fd'], len(fast_shortlist))} candidates..."
        )

    cache: Dict = {}
    fd_ranked = []
    tried = len(unique)
    top_fd_candidates = fast_shortlist[:settings["top_fd"]]

    def _fd_task(layout):
        local_cache = {} if worker_count > 1 else cache
        score, info = _score_layout_by_coarse_heat_solve(
            cfg,
            layout,
            margin_cm,
            coarse_grid_mm,
            local_cache,
            stop_event=stop_event,
            max_iter=1800,
            tolerance_c=0.035,
        )
        return score, layout, info

    if worker_count > 1 and len(top_fd_candidates) > 1:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_fd_task, layout) for layout in top_fd_candidates]
            for fut in concurrent.futures.as_completed(futures):
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Optimization was cancelled.")
                score, layout, info = fut.result()
                tried += 1
                completed += 1
                fd_ranked.append((score, layout, info))
                if progress_callback is not None and (completed == 1 or completed % 5 == 0 or completed == len(top_fd_candidates)):
                    progress_callback(
                        f"Coarse heat solve {completed}/{len(top_fd_candidates)} using {worker_count} workers: "
                        f"current best max≈{min(x[2].get('plate_max_c', x[0]) for x in fd_ranked):.1f} °C"
                    )
    else:
        for idx, layout in enumerate(top_fd_candidates, 1):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Optimization was cancelled.")

            score, info = _score_layout_by_coarse_heat_solve(
                cfg,
                layout,
                margin_cm,
                coarse_grid_mm,
                cache,
                stop_event=stop_event,
                max_iter=1800,
                tolerance_c=0.035,
            )
            tried += 1
            fd_ranked.append((score, layout, info))

            if progress_callback is not None and (idx == 1 or idx % 5 == 0 or idx == len(top_fd_candidates)):
                progress_callback(
                    f"Coarse heat solve {idx}/{len(top_fd_candidates)}: "
                    f"current best max≈{min(x[2].get('plate_max_c', x[0]) for x in fd_ranked):.1f} °C"
                )

    fd_ranked = sorted(fd_ranked, key=lambda x: x[0])
    if not fd_ranked:
        raise ValueError("Optimizer could not evaluate any layouts.")

    refine_starts = fd_ranked[:min(settings["starts_to_refine"], len(fd_ranked))]

    best_layout = [tuple(p) for p in refine_starts[0][1]]
    best_score = float(refine_starts[0][0])
    best_info = dict(refine_starts[0][2])

    min_dim = min(plate_x_cm, plate_y_cm)
    step_sizes = [max(0.25, min_dim * f) for f in settings["step_factor"]]

    for start_i, (start_score, start_layout, start_info) in enumerate(refine_starts, 1):
        if progress_callback is not None:
            progress_callback(
                f"Refining candidate {start_i}/{len(refine_starts)} "
                f"starting max≈{start_info.get('plate_max_c', start_score):.1f} °C"
            )

        layout, score, info, evals = _locally_refine_layout_with_heat_solve(
            cfg=cfg,
            start_layout=[tuple(p) for p in start_layout],
            margin_cm=margin_cm,
            grid_mm=coarse_grid_mm,
            cache=cache,
            step_sizes_cm=step_sizes,
            stop_event=stop_event,
            progress_callback=progress_callback,
            label=f"candidate {start_i}",
            max_passes_per_step=settings["passes"],
        )
        tried += evals

        if score < best_score:
            best_layout = layout
            best_score = score
            best_info = info

    # Final fine-grid check and small final refinement.
    if progress_callback is not None:
        progress_callback("Running finer-grid verification/refinement...")

    fine_cache: Dict = {}
    fine_steps = [max(0.20, min_dim * 0.025), max(0.15, min_dim * 0.012), 0.35]
    best_layout, fine_score, fine_info, fine_evals = _locally_refine_layout_with_heat_solve(
        cfg=cfg,
        start_layout=best_layout,
        margin_cm=margin_cm,
        grid_mm=fine_grid_mm,
        cache=fine_cache,
        step_sizes_cm=fine_steps,
        stop_event=stop_event,
        progress_callback=progress_callback,
        label="fine",
        max_passes_per_step=1,
    )
    tried += fine_evals

    details = {
        "depth": depth,
        "coarse_grid_mm": coarse_grid_mm,
        "fine_grid_mm": fine_grid_mm,
        "valid_layouts_generated": len(unique),
        "coarse_fd_candidates": len(fd_ranked),
        "final_plate_max_c": fine_info.get("plate_max_c"),
        "final_hottest_resistor_c": fine_info.get("hottest_resistor_c"),
        "final_plate_avg_c": fine_info.get("plate_avg_c"),
        "final_grid_cells": fine_info.get("grid_cells"),
        "score": fine_score,
        "worker_count": worker_count,
    }

    if progress_callback is not None:
        progress_callback(
            f"Optimizer complete. Fine-grid max≈{fine_info.get('plate_max_c', fine_score):.1f} °C. "
            f"Tried/evaluated about {tried} layouts/moves."
        )

    return best_layout, fine_score, tried, details


class ThermalPlateGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Thermal Plate Simulator v8")
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
        self._preview_after_id = None
        self._updating_resistor_fields = False
        self._live_preview_ready = False
        self._optimization_running = False

        self.vmin: Optional[float] = None
        self.vmax: Optional[float] = None

        self._build_ui()
        self._refresh_resistor_tree()
        self._update_estimated_h()
        self._update_material_field_states()
        self._setup_live_traces()
        self._live_preview_ready = True
        self._draw_layout_preview()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left side is scrollable. v5 added enough controls that the old fixed
        # panel could run below the bottom of smaller screens.
        left_outer = ttk.Frame(self)
        left_outer.grid(row=0, column=0, sticky="ns")
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(left_outer, width=430, highlightthickness=0)
        self.left_canvas.grid(row=0, column=0, sticky="ns")

        left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=self.left_canvas.yview)
        left_scroll.grid(row=0, column=1, sticky="ns")
        self.left_canvas.configure(yscrollcommand=left_scroll.set)

        left = ttk.Frame(self.left_canvas, padding=8)
        self.left_window_id = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _update_scrollregion(event=None):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def _fit_inner_width(event):
            # Keep the inner frame equal to the canvas width so entries/buttons
            # do not get clipped horizontally.
            self.left_canvas.itemconfigure(self.left_window_id, width=event.width)

        def _wheel_windows(event):
            # Windows/macOS mousewheel. Negative delta scrolls down.
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _wheel_linux_up(event):
            self.left_canvas.yview_scroll(-3, "units")

        def _wheel_linux_down(event):
            self.left_canvas.yview_scroll(3, "units")

        def _bind_mousewheel(event):
            self.left_canvas.bind_all("<MouseWheel>", _wheel_windows)
            self.left_canvas.bind_all("<Button-4>", _wheel_linux_up)
            self.left_canvas.bind_all("<Button-5>", _wheel_linux_down)

        def _unbind_mousewheel(event):
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")

        left.bind("<Configure>", _update_scrollregion)
        self.left_canvas.bind("<Configure>", _fit_inner_width)
        self.left_canvas.bind("<Enter>", _bind_mousewheel)
        self.left_canvas.bind("<Leave>", _unbind_mousewheel)

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

        self._entry_row(plate_frame, 0, "Plate width/X cm", self.plate_length_var)
        self._entry_row(plate_frame, 1, "Plate height/Y cm", self.plate_width_var)
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

        self.k_entry = self._entry_row(plate_frame, 4, "k W/mK", self.k_var)
        self.rho_entry = self._entry_row(plate_frame, 5, "Density kg/m³", self.rho_var)
        self.cp_entry = self._entry_row(plate_frame, 6, "Heat cap J/kgK", self.cp_var)

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

        bank_frame = ttk.LabelFrame(resistor_frame, text="Even-spread bank", padding=6)
        bank_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.bank_count_var = tk.StringVar(value="4")
        self.bank_power_var = tk.StringVar(value="50")
        self.bank_len_var = tk.StringVar(value="50")
        self.bank_wid_var = tk.StringVar(value="20")
        self.bank_margin_var = tk.StringVar(value="1")
        self.optimizer_depth_var = tk.StringVar(value="Deep")
        self.optimizer_grid_var = tk.StringVar(value="12")
        bank_fields = [("Count", self.bank_count_var), ("W each", self.bank_power_var), ("Len mm", self.bank_len_var), ("Wid mm", self.bank_wid_var), ("Margin cm", self.bank_margin_var)]
        for bi, (blabel, bvar) in enumerate(bank_fields):
            ttk.Label(bank_frame, text=blabel).grid(row=bi // 2, column=(bi % 2) * 2, sticky="w", pady=2)
            ttk.Entry(bank_frame, textvariable=bvar, width=8).grid(row=bi // 2, column=(bi % 2) * 2 + 1, sticky="ew", pady=2)

        ttk.Label(bank_frame, text="Opt depth").grid(row=3, column=0, sticky="w", pady=2)
        opt_depth = ttk.Combobox(
            bank_frame,
            textvariable=self.optimizer_depth_var,
            values=["Balanced", "Deep", "Extreme"],
            state="readonly",
            width=10,
        )
        opt_depth.grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(bank_frame, text="Opt grid mm").grid(row=3, column=2, sticky="w", pady=2)
        ttk.Entry(bank_frame, textvariable=self.optimizer_grid_var, width=8).grid(row=3, column=3, sticky="ew", pady=2)

        ttk.Button(bank_frame, text="Create even bank", command=self._create_even_bank).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(bank_frame, text="Deep optimize", command=self._start_optimize_locations).grid(row=4, column=2, columnspan=2, sticky="ew", pady=(6, 0), padx=(4, 0))

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
        ttk.Button(run_frame, text="Field help / value meanings", command=self._show_help_window).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

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
        self.ax.set_title("Layout preview")
        self.ax.set_xlabel("x position / plate width, cm")
        self.ax.set_ylabel("y position / plate height, cm")
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
        if name in MATERIALS and name != "custom":
            m = MATERIALS[name]
            self.k_var.set(str(m["k"]))
            self.rho_var.set(str(m["rho"]))
            self.cp_var.set(str(m["cp"]))
        self._update_material_field_states()
        self._schedule_preview_update()

    def _update_material_field_states(self):
        if not all(hasattr(self, attr) for attr in ("k_entry", "rho_entry", "cp_entry")):
            return
        state = "normal" if self.material_var.get() == "custom" else "disabled"
        for entry in (self.k_entry, self.rho_entry, self.cp_entry):
            try:
                entry.configure(state=state)
            except Exception:
                pass

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
            self._updating_resistor_fields = True
            try:
                self.r_name_var.set(r.name)
                self.r_power_var.set(f"{r.power_w:g}")
                self.r_x_var.set(f"{r.center_x_cm:g}")
                self.r_y_var.set(f"{r.center_y_cm:g}")
                self.r_len_var.set(f"{r.length_mm:g}")
                self.r_wid_var.set(f"{r.width_mm:g}")
            finally:
                self._updating_resistor_fields = False

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
            self._schedule_preview_update()
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
            self._schedule_preview_update()
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
            self._schedule_preview_update()

    def _spread_four_example(self):
        self.resistors = [
            Resistor("R1", 50.0, -5.0, -5.0, 50.0, 20.0),
            Resistor("R2", 50.0,  5.0, -5.0, 50.0, 20.0),
            Resistor("R3", 50.0, -5.0,  5.0, 50.0, 20.0),
            Resistor("R4", 50.0,  5.0,  5.0, 50.0, 20.0),
        ]
        self._refresh_resistor_tree()
        self._schedule_preview_update()
        self._set_status("Loaded example with four 50 W resistors spread around the center.")

    def _create_even_bank(self):
        try:
            count = int(parse_float(self.bank_count_var.get(), "Bank count", 1))
            if count < 1:
                raise ValueError("Bank count must be at least 1.")
            power_each = parse_float(self.bank_power_var.get(), "Bank power each", 0.0)
            length_mm = parse_float(self.bank_len_var.get(), "Bank resistor length", 0.1)
            width_mm = parse_float(self.bank_wid_var.get(), "Bank resistor width", 0.1)
            margin_cm = parse_float(self.bank_margin_var.get(), "Bank margin", 0.0)
            plate_x = parse_float(self.plate_length_var.get(), "Plate width/X", 0.1)
            plate_y = parse_float(self.plate_width_var.get(), "Plate height/Y", 0.1)
            positions = evenly_spaced_positions(count, plate_x, plate_y, length_mm, width_mm, margin_cm)
            self.resistors = [Resistor(f"R{i + 1}", power_each, x, y, length_mm, width_mm) for i, (x, y) in enumerate(positions)]
            self._refresh_resistor_tree()
            self._schedule_preview_update()
            self._set_status(f"Created {count} evenly-spread resistors. Run simulation to calculate temperatures.")
        except Exception as e:
            messagebox.showerror("Even-spread bank failed", str(e))

    def _start_optimize_locations(self):
        if self._optimization_running:
            messagebox.showinfo("Optimizer", "Optimization is already running.")
            return
        try:
            cfg = self._read_config()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return
        if len(cfg.resistors) < 2:
            messagebox.showinfo("Optimizer", "Add at least two resistors to optimize spacing.")
            return
        self._optimization_running = True
        self.stop_event.clear()
        self.cancel_button.configure(state="normal")
        self._set_status(
            f"Deep optimizer running. Depth={self.optimizer_depth_var.get()}, "
            f"optimizer grid={self.optimizer_grid_var.get()} mm.\n"
            "This now uses actual coarse heat solves during optimization, so it can take longer."
        )
        def worker():
            try:
                best_positions, best_temp, tried, details = self._optimize_locations_worker(cfg)
                self.msg_queue.put(("opt_done", (best_positions, best_temp, tried, details)))
            except Exception as e:
                tb = traceback.format_exc()
                self.msg_queue.put(("opt_error", f"{e}\n\n{tb}"))
        threading.Thread(target=worker, daemon=True).start()

    def _optimize_locations_worker(self, cfg: PlateConfig):
        try:
            margin_cm = parse_float(self.bank_margin_var.get(), "Bank margin", 0.0)
        except Exception:
            margin_cm = 1.0

        try:
            opt_grid_mm = parse_float(self.optimizer_grid_var.get(), "Optimizer grid", 3.0)
        except Exception:
            opt_grid_mm = 12.0

        depth = self.optimizer_depth_var.get()

        positions, best_score, tried, details = optimize_layout_deep(
            cfg=cfg,
            margin_cm=margin_cm,
            optimizer_grid_mm=opt_grid_mm,
            depth=depth,
            progress_callback=lambda msg: self.msg_queue.put(("progress", msg)),
            stop_event=self.stop_event,
        )

        return positions, best_score, tried, details

    def _apply_optimized_positions(self, positions: List[Tuple[float, float]], best_temp: float, tried: int, details: Optional[Dict] = None):
        for r, (x, y) in zip(self.resistors, positions):
            r.center_x_cm = x
            r.center_y_cm = y
        self._refresh_resistor_tree()
        self._schedule_preview_update()
        details = details or {}
        msg = [
            f"Deep optimizer finished. Tried/evaluated about {tried} layouts/moves.",
            f"Best optimizer score: {best_temp:.3f}  lower is better.",
        ]
        if details:
            msg.append(f"Depth: {details.get('depth')}")
            msg.append(f"Generated valid layouts: {details.get('valid_layouts_generated')}")
            msg.append(f"Coarse FD candidates: {details.get('coarse_fd_candidates')}")
            msg.append(f"Optimizer coarse grid: {details.get('coarse_grid_mm')} mm")
            msg.append(f"Optimizer fine grid: {details.get('fine_grid_mm')} mm")
            if details.get("final_plate_max_c") is not None:
                msg.append(f"Fine-grid estimated max: {details.get('final_plate_max_c'):.1f} °C")
            if details.get("final_hottest_resistor_c") is not None:
                msg.append(f"Fine-grid hottest resistor footprint: {details.get('final_hottest_resistor_c'):.1f} °C")
        msg.append("")
        msg.append("Positions have been applied. Run the full simulation for the final detailed result.")
        msg.append("Use Extreme if this still looks too shallow, but it can take much longer.")
        self._set_status("\n".join(msg))

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
                elif msg_type == "opt_done":
                    self._optimization_running = False
                    self.cancel_button.configure(state="disabled")
                    if len(payload) == 4:
                        positions, best_temp, tried, details = payload
                    else:
                        positions, best_temp, tried = payload
                        details = {}
                    self._apply_optimized_positions(positions, best_temp, tried, details)
                elif msg_type == "opt_error":
                    self._optimization_running = False
                    self.cancel_button.configure(state="disabled")
                    self._append_status("Optimizer error:\n" + str(payload))
                    messagebox.showerror("Optimizer error", str(payload).split("\n\n")[0])
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

            extent = [
                -cfg.plate_length_cm / 2.0,
                 cfg.plate_length_cm / 2.0,
                -cfg.plate_width_cm / 2.0,
                 cfg.plate_width_cm / 2.0,
            ]

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
            self.ax.set_xlabel("x position / plate width, cm")
            self.ax.set_ylabel("y position / plate height, cm")
            self.ax.grid(True, alpha=0.25)

            # Avoid tight_layout here. On some Windows/Matplotlib combinations
            # it can trigger deep recursion during rapid redraws.
            self.fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.92)
            self.canvas.draw_idle()

            self.snapshot_label_var.set(f"{idx + 1}/{len(self.result.snapshots)}: {snap.label}")
            self._current_snapshot_idx = idx
        finally:
            self._drawing_snapshot = False

    def _setup_live_traces(self):
        vars_to_watch = [
            self.plate_length_var, self.plate_width_var, self.plate_thickness_var,
            self.ambient_var, self.h_var, self.grid_var, self.initial_temp_var,
            self.max_time_var, self.snapshot_every_var, self.material_var,
            self.k_var, self.rho_var, self.cp_var,
            self.advanced_cooling_var, self.orientation_var, self.environment_var,
            self.clearance_var, self.surface_var, self.air_movement_var,
            self.hot_air_path_var, self.blockage_var,
        ]
        for var in vars_to_watch:
            try:
                var.trace_add("write", lambda *args: self._schedule_preview_update())
            except Exception:
                pass
        for var in [self.r_name_var, self.r_power_var, self.r_x_var, self.r_y_var, self.r_len_var, self.r_wid_var]:
            try:
                var.trace_add("write", lambda *args: self._live_resistor_edit_changed())
            except Exception:
                pass

    def _live_resistor_edit_changed(self):
        if self._updating_resistor_fields:
            return
        sel = self.res_tree.selection() if hasattr(self, "res_tree") else []
        if sel:
            try:
                idx = int(sel[0])
                if 0 <= idx < len(self.resistors):
                    self.resistors[idx] = self._read_resistor_fields()
                    self._refresh_resistor_tree()
                    self.res_tree.selection_set(str(idx))
            except Exception:
                pass
        self._schedule_preview_update()

    def _schedule_preview_update(self):
        if not getattr(self, "_live_preview_ready", False):
            return
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.after(180, self._draw_layout_preview)

    def _draw_layout_preview(self):
        self._preview_after_id = None
        if self._drawing_snapshot:
            return
        try:
            plate_x = parse_float(self.plate_length_var.get(), "Plate width/X", 0.1)
            plate_y = parse_float(self.plate_width_var.get(), "Plate height/Y", 0.1)
        except Exception:
            return
        self.result = None
        self._current_snapshot_idx = None
        self._ignore_slider_callback = True
        try:
            self.time_slider.configure(from_=0, to=0)
            self.time_slider.set(0)
        except Exception:
            pass
        finally:
            self._ignore_slider_callback = False
        self.snapshot_label_var.set("Layout preview")
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None
        plate_rect = Rectangle((-plate_x / 2.0, -plate_y / 2.0), plate_x, plate_y, fill=False, linewidth=2)
        self.ax.add_patch(plate_rect)
        for r in self.resistors:
            l_cm = r.length_mm / 10.0
            w_cm = r.width_mm / 10.0
            x0 = r.center_x_cm - l_cm / 2.0
            y0 = r.center_y_cm - w_cm / 2.0
            in_bounds = (x0 >= -plate_x / 2.0 and x0 + l_cm <= plate_x / 2.0 and y0 >= -plate_y / 2.0 and y0 + w_cm <= plate_y / 2.0)
            rect = Rectangle((x0, y0), l_cm, w_cm, fill=True, alpha=0.25 if in_bounds else 0.55, linewidth=2)
            self.ax.add_patch(rect)
            self.ax.text(r.center_x_cm, r.center_y_cm, r.name, ha="center", va="center", fontsize=9)
        pad = max(1.0, 0.08 * max(plate_x, plate_y))
        self.ax.set_xlim(-plate_x / 2.0 - pad, plate_x / 2.0 + pad)
        self.ax.set_ylim(-plate_y / 2.0 - pad, plate_y / 2.0 + pad)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Layout preview — run simulation for heat map")
        self.ax.set_xlabel("x position / plate width, cm")
        self.ax.set_ylabel("y position / plate height, cm")
        self.ax.grid(True, alpha=0.25)
        self.fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.92)
        self.canvas.draw_idle()

    def _show_help_window(self):
        win = tk.Toplevel(self)
        win.title("Thermal simulator field help")
        win.geometry("720x620")
        txt = tk.Text(win, wrap="word", padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", FIELD_HELP_TEXT)
        txt.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)

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
        self._update_material_field_states()
        self._schedule_preview_update()

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
