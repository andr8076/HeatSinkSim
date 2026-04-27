#!/usr/bin/env python3
"""
thermal_plate_sim_v2.py

Passive flat-plate thermal simulator with:
  - Steady-state heatmap
  - Optional transient / warm-up heatmaps over time
  - Multiple rectangular resistors / heat sources
  - Rough warm-up behavior estimate
  - CSV and JSON outputs

Model:
  Thin metal plate, 2D heat spreading.
  Heat leaves both large faces by convection.
  Resistors inject heat over their contact footprints.

Important:
  This is an engineering approximation, not a certified safety calculation.
  Use real temperature measurement and a cutoff for real hardware.
  The simulated temperature is plate/contact temperature, not guaranteed internal resistor temperature.

Dependencies:
  python3 -m pip install numpy matplotlib
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy")
    print("Install with: python3 -m pip install numpy matplotlib")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    print("Missing dependency: matplotlib")
    print("Install with: python3 -m pip install numpy matplotlib")
    sys.exit(1)


MATERIALS = {
    "aluminium": {"k": 205.0, "rho": 2700.0, "cp": 900.0},
    "aluminum":  {"k": 205.0, "rho": 2700.0, "cp": 900.0},
    "copper":    {"k": 385.0, "rho": 8960.0, "cp": 385.0},
    "steel":     {"k": 45.0,  "rho": 7850.0, "cp": 470.0},
    "iron":      {"k": 80.0,  "rho": 7870.0, "cp": 450.0},
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


@dataclass
class TransientConfig:
    enabled: bool
    initial_plate_temp_c: float
    total_time_s: float
    snapshot_times_s: List[float]
    max_steps_without_confirm: int = 250000


def ask_float(prompt: str, default: Optional[float] = None, min_value: Optional[float] = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip().replace(",", ".")
        if raw == "" and default is not None:
            value = float(default)
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue

        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue

        return value


def ask_int(prompt: str, default: Optional[int] = None, min_value: Optional[int] = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue

        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue

        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    yes = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{yes}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes", "j", "ja"):
            return True
        if raw in ("n", "no", "nej"):
            return False
        print("Please answer yes or no.")


def parse_time_list(raw: str) -> List[float]:
    """
    Accepts:
      30s, 1m, 5m, 15m, 1h
      or plain numbers, interpreted as minutes.
    """
    out: List[float] = []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        m = re.fullmatch(r"([0-9]+(?:[.,][0-9]+)?)(\s*[smhSMH]?)", part)
        if not m:
            raise ValueError(f"Could not understand time: {part}")

        value = float(m.group(1).replace(",", "."))
        unit = m.group(2).strip().lower()

        if unit == "s":
            seconds = value
        elif unit == "h":
            seconds = value * 3600.0
        else:
            # default and "m" are minutes
            seconds = value * 60.0

        if seconds > 0:
            out.append(seconds)

    return sorted(set(out))


def format_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    return f"{hours:.2f} h"


def safe_filename_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60.0:.1f}min".replace(".", "p")
    return f"{seconds / 3600.0:.2f}h".replace(".", "p")


def choose_material() -> Tuple[str, float, float, float]:
    print("\nMaterial presets:")
    print("  1) aluminium / aluminum")
    print("  2) copper")
    print("  3) steel")
    print("  4) custom")

    choice = input("Choose material [1]: ").strip()
    if choice == "":
        choice = "1"

    if choice == "1":
        m = MATERIALS["aluminium"]
        return "aluminium", m["k"], m["rho"], m["cp"]
    if choice == "2":
        m = MATERIALS["copper"]
        return "copper", m["k"], m["rho"], m["cp"]
    if choice == "3":
        m = MATERIALS["steel"]
        return "steel", m["k"], m["rho"], m["cp"]

    print("\nCustom material:")
    print("Thermal conductivity examples:")
    print("  aluminium ≈ 205 W/mK")
    print("  copper    ≈ 385 W/mK")
    print("  steel     ≈ 45 W/mK")
    k = ask_float("Thermal conductivity k, W/mK", 205.0, 0.1)
    rho = ask_float("Density, kg/m³", 2700.0, 1.0)
    cp = ask_float("Specific heat capacity, J/kgK", 900.0, 1.0)
    return "custom", k, rho, cp


def make_config_interactive() -> PlateConfig:
    print("\n=== Plate ===")
    plate_length_cm = ask_float("Plate length, cm", 25.0, 0.1)
    plate_width_cm = ask_float("Plate width, cm", 25.0, 0.1)
    plate_thickness_mm = ask_float("Plate thickness, mm", 5.0, 0.1)

    material_name, k, rho, cp = choose_material()

    print("\n=== Cooling ===")
    ambient_c = ask_float("Ambient air temperature, °C", 25.0)

    print("\nConvection h guide:")
    print("  5  = weak passive, bad orientation/still air")
    print("  7  = normal passive estimate")
    print("  10 = good passive, vertical plate/open air")
    convection_h = ask_float("Convection coefficient h, W/m²K", 7.0, 0.1)

    print("\n=== Simulation grid ===")
    print("Smaller grid = better detail, slower transient simulation.")
    print("Good starting point:")
    print("  2.5 mm = nice detail, slower transient")
    print("  5.0 mm = good practical default")
    print("  10 mm  = quick rough testing")
    grid_mm = ask_float("Grid cell size, mm", 5.0, 0.5)

    print("\n=== Resistors / heat sources ===")
    n = ask_int("Number of resistors", 1, 1)
    resistors: List[Resistor] = []

    print("\nCoordinate system:")
    print("  Plate center is x=0, y=0.")
    print("  x goes left/right, y goes up/down.")
    print("  Example: one centered resistor => x=0, y=0.")

    for i in range(n):
        print(f"\nResistor {i + 1}:")
        name = input(f"Name [R{i + 1}]: ").strip() or f"R{i + 1}"
        power_w = ask_float("Power, W", 50.0, 0.0)
        length_mm = ask_float("Contact length touching plate, mm", 50.0, 0.1)
        width_mm = ask_float("Contact width touching plate, mm", 20.0, 0.1)
        center_x_cm = ask_float("Center x position, cm", 0.0)
        center_y_cm = ask_float("Center y position, cm", 0.0)

        resistors.append(
            Resistor(
                name=name,
                power_w=power_w,
                center_x_cm=center_x_cm,
                center_y_cm=center_y_cm,
                length_mm=length_mm,
                width_mm=width_mm,
            )
        )

    return PlateConfig(
        plate_length_cm=plate_length_cm,
        plate_width_cm=plate_width_cm,
        plate_thickness_mm=plate_thickness_mm,
        material_name=material_name,
        thermal_conductivity_w_mk=k,
        density_kg_m3=rho,
        heat_capacity_j_kgk=cp,
        ambient_c=ambient_c,
        convection_h_w_m2k=convection_h,
        grid_mm=grid_mm,
        resistors=resistors,
    )


def make_transient_config_interactive(cfg: PlateConfig) -> TransientConfig:
    if not ask_yes_no("\nRun transient / time-based warm-up simulation?", True):
        return TransientConfig(
            enabled=False,
            initial_plate_temp_c=cfg.ambient_c,
            total_time_s=0.0,
            snapshot_times_s=[],
        )

    print("\n=== Transient simulation ===")
    initial_temp = ask_float("Initial plate temperature, °C", cfg.ambient_c)

    print("\nSnapshot examples:")
    print("  30s, 1m, 5m, 15m, 30m, 1h")
    print("Plain numbers are interpreted as minutes.")
    while True:
        raw = input("Times to save heatmaps [1m, 5m, 15m, 30m, 1h]: ").strip()
        if raw == "":
            raw = "1m, 5m, 15m, 30m, 1h"
        try:
            times = parse_time_list(raw)
            if not times:
                raise ValueError("No valid times entered.")
            break
        except ValueError as e:
            print(e)

    return TransientConfig(
        enabled=True,
        initial_plate_temp_c=initial_temp,
        total_time_s=max(times),
        snapshot_times_s=times,
    )


def load_config(path: Path) -> PlateConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    resistors = [Resistor(**r) for r in data["resistors"]]
    data["resistors"] = resistors
    return PlateConfig(**data)


def save_config(cfg: PlateConfig, path: Path) -> None:
    data = asdict(cfg)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_grid(cfg: PlateConfig):
    L = cfg.plate_length_cm / 100.0
    W = cfg.plate_width_cm / 100.0
    dx = cfg.grid_mm / 1000.0
    dy = dx

    nx = max(3, int(round(L / dx)))
    ny = max(3, int(round(W / dy)))

    # Recompute dx/dy so the grid exactly spans the plate.
    dx = L / nx
    dy = W / ny

    x = (np.arange(nx) + 0.5) * dx - L / 2.0
    y = (np.arange(ny) + 0.5) * dy - W / 2.0

    return x, y, dx, dy


def add_heat_sources(cfg: PlateConfig, x, y, dx, dy):
    nx = len(x)
    ny = len(y)
    q = np.zeros((nx, ny), dtype=float)  # W/m² applied to plate top surface

    X, Y = np.meshgrid(x, y, indexing="ij")
    cell_area = dx * dy

    resistor_masks = []

    for r in cfg.resistors:
        cx = r.center_x_cm / 100.0
        cy = r.center_y_cm / 100.0
        half_l = (r.length_mm / 1000.0) / 2.0
        half_w = (r.width_mm / 1000.0) / 2.0

        mask = (
            (X >= cx - half_l) &
            (X <= cx + half_l) &
            (Y >= cy - half_w) &
            (Y <= cy + half_w)
        )

        # If the grid is too coarse and misses the resistor, force nearest cell.
        if not np.any(mask):
            ix = int(np.argmin(np.abs(x - cx)))
            iy = int(np.argmin(np.abs(y - cy)))
            mask[ix, iy] = True

        covered_area = np.count_nonzero(mask) * cell_area
        q[mask] += r.power_w / covered_area
        resistor_masks.append((r, mask, covered_area))

    return q, resistor_masks


def solve_steady_state(cfg: PlateConfig, max_iter: int = 30000, tolerance_c: float = 0.0005):
    """
    Solves steady-state thin-plate equation:

        k*t*laplacian(theta) + q - 2*h*theta = 0

    theta is temperature rise above ambient.
    q is W/m² heat flux inserted into the plate.
    2*h because both large faces lose heat to air.

    Edges use a zero-lateral-flux approximation.
    """
    x, y, dx, dy = build_grid(cfg)
    q, resistor_masks = add_heat_sources(cfg, x, y, dx, dy)

    k = cfg.thermal_conductivity_w_mk
    t = cfg.plate_thickness_mm / 1000.0
    h = cfg.convection_h_w_m2k

    theta = np.zeros_like(q)

    ax = k * t / (dx * dx)
    ay = k * t / (dy * dy)
    center = 2.0 * ax + 2.0 * ay + 2.0 * h

    converged = False
    final_diff = None

    for it in range(1, max_iter + 1):
        p = np.pad(theta, ((1, 1), (1, 1)), mode="edge")
        left = p[:-2, 1:-1]
        right = p[2:, 1:-1]
        down = p[1:-1, :-2]
        up = p[1:-1, 2:]

        new_theta = (ax * (left + right) + ay * (down + up) + q) / center

        final_diff = float(np.max(np.abs(new_theta - theta)))
        theta = new_theta

        if final_diff < tolerance_c:
            converged = True
            break

    temp_c = cfg.ambient_c + theta
    return {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "q": q,
        "temp_c": temp_c,
        "theta_c": theta,
        "resistor_masks": resistor_masks,
        "iterations": it,
        "converged": converged,
        "final_diff": final_diff,
    }


def transient_stability_dt(cfg: PlateConfig, dx: float, dy: float) -> float:
    """
    Explicit finite difference stability estimate.

    Equation:
      dtheta/dt = alpha*laplacian(theta) + source - beta*theta

    Stable-ish explicit dt:
      dt < 1 / (2*alpha*(1/dx² + 1/dy²) + beta)
    """
    k = cfg.thermal_conductivity_w_mk
    rho = cfg.density_kg_m3
    cp = cfg.heat_capacity_j_kgk
    t = cfg.plate_thickness_mm / 1000.0
    h = cfg.convection_h_w_m2k

    alpha = k / (rho * cp)
    beta = 2.0 * h / (rho * cp * t)

    denom = 2.0 * alpha * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))) + beta
    return 0.45 / denom


def run_transient(cfg: PlateConfig, tr: TransientConfig, base_result: dict, output_dir: Path):
    """
    Explicit transient solver.

    It simulates real elapsed time from initial plate temperature toward steady-state.
    For fine grids, this may require many time steps.
    """
    x = base_result["x"]
    y = base_result["y"]
    dx = base_result["dx"]
    dy = base_result["dy"]
    q = base_result["q"]
    resistor_masks = base_result["resistor_masks"]

    rho = cfg.density_kg_m3
    cp = cfg.heat_capacity_j_kgk
    t = cfg.plate_thickness_mm / 1000.0
    k = cfg.thermal_conductivity_w_mk
    h = cfg.convection_h_w_m2k

    alpha = k / (rho * cp)
    beta = 2.0 * h / (rho * cp * t)
    source = q / (rho * cp * t)

    dt = transient_stability_dt(cfg, dx, dy)
    steps = int(math.ceil(tr.total_time_s / dt))
    dt = tr.total_time_s / steps  # exact landing on total time

    print("\nTransient simulation details:")
    print(f"  Grid: {len(x)} × {len(y)} cells")
    print(f"  Stable timestep used: {dt:.4f} s")
    print(f"  Number of timesteps: {steps}")

    if steps > tr.max_steps_without_confirm:
        print("\nWarning:")
        print(f"  This transient simulation needs about {steps} timesteps.")
        print("  It may take a while.")
        print("  To speed it up, increase grid cell size, for example 5 mm or 10 mm.")
        if not ask_yes_no("Continue anyway?", True):
            print("Transient simulation skipped.")
            return None

    theta = np.full_like(q, tr.initial_plate_temp_c - cfg.ambient_c, dtype=float)

    snapshot_times = sorted(tr.snapshot_times_s)
    snapshot_index = 0
    snapshots: Dict[float, np.ndarray] = {}
    rows = []

    cell_area = dx * dy

    def store_snapshot(time_s: float):
        temp = cfg.ambient_c + theta
        snap = temp.copy()
        snapshots[time_s] = snap

        row = {
            "time_s": time_s,
            "time_label": format_time(time_s),
            "avg_temp_c": float(np.mean(snap)),
            "max_temp_c": float(np.max(snap)),
            "min_temp_c": float(np.min(snap)),
            "convective_loss_w": float(np.sum(2.0 * h * (snap - cfg.ambient_c) * cell_area)),
        }

        for r, mask, covered_area in resistor_masks:
            row[f"{r.name}_avg_temp_c"] = float(np.mean(snap[mask]))
            row[f"{r.name}_max_temp_c"] = float(np.max(snap[mask]))

        rows.append(row)

    current_time = 0.0

    for step in range(1, steps + 1):
        p = np.pad(theta, ((1, 1), (1, 1)), mode="edge")
        left = p[:-2, 1:-1]
        right = p[2:, 1:-1]
        down = p[1:-1, :-2]
        up = p[1:-1, 2:]

        lap = (
            (left - 2.0 * theta + right) / (dx * dx)
            + (down - 2.0 * theta + up) / (dy * dy)
        )

        theta += dt * (alpha * lap + source - beta * theta)
        current_time = step * dt

        while snapshot_index < len(snapshot_times) and current_time >= snapshot_times[snapshot_index] - 0.5 * dt:
            store_snapshot(snapshot_times[snapshot_index])
            snapshot_index += 1

    # Ensure final requested snapshot exists even with rounding.
    while snapshot_index < len(snapshot_times):
        store_snapshot(snapshot_times[snapshot_index])
        snapshot_index += 1

    # Save transient summary CSV.
    csv_path = output_dir / "transient_summary.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    # Save heatmaps.
    heatmap_paths = []
    for time_s, temp in snapshots.items():
        png_path = output_dir / f"transient_heatmap_{safe_filename_time(time_s)}.png"
        plot_heatmap(
            cfg=cfg,
            result={**base_result, "temp_c": temp},
            output_png=png_path,
            title=f"Plate heat map after {format_time(time_s)}",
        )
        heatmap_paths.append(str(png_path))

    # Save temperature grids for each snapshot.
    for time_s, temp in snapshots.items():
        csv_grid_path = output_dir / f"temperature_grid_{safe_filename_time(time_s)}.csv"
        save_csv_from_temp(base_result["x"], base_result["y"], temp, csv_grid_path)

    transient_json = {
        "enabled": True,
        "initial_plate_temp_c": tr.initial_plate_temp_c,
        "total_time_s": tr.total_time_s,
        "dt_s": dt,
        "steps": steps,
        "snapshots": rows,
        "heatmap_files": heatmap_paths,
    }
    json_path = output_dir / "transient_summary.json"
    json_path.write_text(json.dumps(transient_json, indent=2), encoding="utf-8")

    print("\nTransient files written:")
    print(f"  Transient summary CSV: {csv_path}")
    print(f"  Transient summary JSON: {json_path}")
    for p in heatmap_paths:
        print(f"  Heatmap: {p}")

    return transient_json


def calculate_summary(cfg: PlateConfig, result: dict):
    temp = result["temp_c"]
    theta = result["theta_c"]
    dx = result["dx"]
    dy = result["dy"]
    cell_area = dx * dy

    plate_l_m = cfg.plate_length_cm / 100.0
    plate_w_m = cfg.plate_width_cm / 100.0
    plate_t_m = cfg.plate_thickness_mm / 1000.0

    total_power = sum(r.power_w for r in cfg.resistors)
    top_bottom_area = 2.0 * plate_l_m * plate_w_m
    edge_area = 2.0 * (plate_l_m + plate_w_m) * plate_t_m
    exposed_area_with_edges = top_bottom_area + edge_area

    predicted_avg_rise_simple = total_power / (cfg.convection_h_w_m2k * top_bottom_area)
    conv_loss_grid = float(np.sum(2.0 * cfg.convection_h_w_m2k * theta * cell_area))

    volume = plate_l_m * plate_w_m * plate_t_m
    mass = cfg.density_kg_m3 * volume
    thermal_capacity = mass * cfg.heat_capacity_j_kgk

    tau_s = thermal_capacity / (cfg.convection_h_w_m2k * top_bottom_area)
    t90_s = -math.log(0.10) * tau_s
    t95_s = -math.log(0.05) * tau_s

    resistor_reports = []
    for r, mask, covered_area in result["resistor_masks"]:
        resistor_reports.append({
            "name": r.name,
            "power_w": r.power_w,
            "covered_area_cm2": covered_area * 10000.0,
            "max_temp_c": float(np.max(temp[mask])),
            "avg_temp_c": float(np.mean(temp[mask])),
        })

    return {
        "total_power_w": total_power,
        "plate_area_both_sides_m2": top_bottom_area,
        "plate_area_both_sides_cm2": top_bottom_area * 10000.0,
        "plate_area_with_edges_cm2": exposed_area_with_edges * 10000.0,
        "simple_average_rise_c": predicted_avg_rise_simple,
        "simulated_average_temp_c": float(np.mean(temp)),
        "simulated_max_temp_c": float(np.max(temp)),
        "simulated_min_temp_c": float(np.min(temp)),
        "simulated_max_rise_c": float(np.max(theta)),
        "estimated_convective_loss_w": conv_loss_grid,
        "mass_kg": mass,
        "thermal_capacity_j_per_c": thermal_capacity,
        "tau_s": tau_s,
        "t90_s": t90_s,
        "t95_s": t95_s,
        "resistors": resistor_reports,
    }


def print_summary(cfg: PlateConfig, result: dict, summary: dict):
    print("\n================ STEADY-STATE RESULTS ================")
    print(f"Solver converged: {result['converged']} after {result['iterations']} iterations")
    print(f"Final iteration change: {result['final_diff']:.6f} °C")

    print("\nPlate:")
    print(f"  Size: {cfg.plate_length_cm:g} × {cfg.plate_width_cm:g} cm")
    print(f"  Thickness: {cfg.plate_thickness_mm:g} mm")
    print(f"  Material: {cfg.material_name}")
    print(f"  k: {cfg.thermal_conductivity_w_mk:g} W/mK")
    print(f"  h: {cfg.convection_h_w_m2k:g} W/m²K")
    print(f"  Ambient: {cfg.ambient_c:g} °C")
    print(f"  Area, both large faces: {summary['plate_area_both_sides_cm2']:.0f} cm²")

    print("\nPower:")
    print(f"  Total resistor power: {summary['total_power_w']:.2f} W")
    print(f"  Grid-estimated convection loss: {summary['estimated_convective_loss_w']:.2f} W")

    print("\nTemperatures:")
    print(f"  Simple average plate rise: {summary['simple_average_rise_c']:.1f} °C")
    print(f"  Simulated average temp: {summary['simulated_average_temp_c']:.1f} °C")
    print(f"  Simulated max temp: {summary['simulated_max_temp_c']:.1f} °C")
    print(f"  Simulated min temp: {summary['simulated_min_temp_c']:.1f} °C")

    print("\nAt each resistor footprint:")
    for rr in summary["resistors"]:
        print(f"  {rr['name']}:")
        print(f"    power: {rr['power_w']:.2f} W")
        print(f"    simulated contact area: {rr['covered_area_cm2']:.2f} cm²")
        print(f"    avg temp: {rr['avg_temp_c']:.1f} °C")
        print(f"    max temp: {rr['max_temp_c']:.1f} °C")

    print("\nWarm-up behavior, rough whole-plate estimate:")
    print(f"  Plate mass: {summary['mass_kg']:.3f} kg")
    print(f"  Heat capacity: {summary['thermal_capacity_j_per_c']:.0f} J/°C")
    print(f"  Time constant: {format_time(summary['tau_s'])}")
    print(f"  About 90% of steady temp: {format_time(summary['t90_s'])}")
    print(f"  About 95% of steady temp: {format_time(summary['t95_s'])}")

    max_temp = summary["simulated_max_temp_c"]
    print("\nRisk reading:")
    if max_temp < 70:
        print("  Looks comfortable in this model.")
    elif max_temp < 90:
        print("  Warm/hot, but probably usable if mounted well and kept in open air.")
    elif max_temp < 110:
        print("  Very hot. Use temperature cutoff and careful wiring. Expect burn risk.")
    elif max_temp < 140:
        print("  Risky. This is beyond what I would treat as a casual passive setup.")
    else:
        print("  Dangerous. Do not trust this setup without redesign and real testing.")

    print("\nModel limits:")
    print("  - Assumes uniform passive air cooling h over both large faces.")
    print("  - Ignores radiation, screw contact details, paint/anodizing, and resistor internal temperature.")
    print("  - Edges are treated as laterally insulated; edge cooling is small for thin plates.")
    print("  - Real resistor body may be hotter than the simulated plate contact area.")


def plot_heatmap(cfg: PlateConfig, result: dict, output_png: Path, title: str = "Steady-state plate heat map"):
    temp = result["temp_c"]
    x = result["x"]
    y = result["y"]

    extent = [
        x[0] * 100.0,
        x[-1] * 100.0,
        y[0] * 100.0,
        y[-1] * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        temp.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Temperature, °C")

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
        ax.add_patch(rect)
        ax.text(cx, cy, r.name, ha="center", va="center", fontsize=9)

    max_idx = np.unravel_index(np.argmax(temp), temp.shape)
    max_x = x[max_idx[0]] * 100.0
    max_y = y[max_idx[1]] * 100.0
    max_t = temp[max_idx]
    ax.plot([max_x], [max_y], marker="x", markersize=10)
    ax.text(max_x, max_y, f" max {max_t:.1f}°C", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("x position, cm")
    ax.set_ylabel("y position, cm")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def save_csv_from_temp(x_m, y_m, temp, output_csv: Path):
    x_cm = x_m * 100.0
    y_cm = y_m * 100.0

    with output_csv.open("w", encoding="utf-8") as f:
        f.write("x_cm,y_cm,temp_c\n")
        for ix, xv in enumerate(x_cm):
            for iy, yv in enumerate(y_cm):
                f.write(f"{xv:.6f},{yv:.6f},{temp[ix, iy]:.6f}\n")


def save_csv(cfg: PlateConfig, result: dict, output_csv: Path):
    save_csv_from_temp(result["x"], result["y"], result["temp_c"], output_csv)


def example_config() -> PlateConfig:
    mat = MATERIALS["aluminium"]
    return PlateConfig(
        plate_length_cm=25.0,
        plate_width_cm=25.0,
        plate_thickness_mm=5.0,
        material_name="aluminium",
        thermal_conductivity_w_mk=mat["k"],
        density_kg_m3=mat["rho"],
        heat_capacity_j_kgk=mat["cp"],
        ambient_c=25.0,
        convection_h_w_m2k=7.0,
        grid_mm=5.0,
        resistors=[
            Resistor(
                name="R1",
                power_w=50.0,
                center_x_cm=0.0,
                center_y_cm=0.0,
                length_mm=50.0,
                width_mm=20.0,
            )
        ],
    )


def main():
    print("====================================================")
    print(" Passive Plate Thermal Simulator v2")
    print("====================================================")
    print("Steady-state heatmap + optional time-based warm-up heatmaps.")

    cfg: PlateConfig

    print("\nStart options:")
    print("  1) Enter values manually")
    print("  2) Use example: 25×25 cm × 5 mm aluminium plate, 50 W resistor")
    print("  3) Load JSON config")
    choice = input("Choose [1]: ").strip() or "1"

    if choice == "2":
        cfg = example_config()
    elif choice == "3":
        path = Path(input("Config JSON path: ").strip()).expanduser()
        cfg = load_config(path)
    else:
        cfg = make_config_interactive()

    tr = make_transient_config_interactive(cfg)

    output_dir = Path("thermal_output")
    output_dir.mkdir(exist_ok=True)

    if ask_yes_no("\nSave this setup as JSON config?", True):
        save_path = output_dir / "last_config.json"
        save_config(cfg, save_path)
        print(f"Saved config: {save_path}")

    print("\nSolving steady-state final temperature...")
    result = solve_steady_state(cfg)
    summary = calculate_summary(cfg, result)

    print_summary(cfg, result, summary)

    steady_png_path = output_dir / "steady_state_heatmap.png"
    steady_csv_path = output_dir / "steady_state_temperature_grid.csv"
    summary_path = output_dir / "steady_state_summary.json"

    plot_heatmap(cfg, result, steady_png_path, title="Steady-state plate heat map")
    save_csv(cfg, result, steady_csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSteady-state files written:")
    print(f"  Heat map: {steady_png_path}")
    print(f"  Temperature grid CSV: {steady_csv_path}")
    print(f"  Summary JSON: {summary_path}")

    if tr.enabled:
        run_transient(cfg, tr, result, output_dir)

    if ask_yes_no("\nOpen steady-state heat map window now?", False):
        img = plt.imread(steady_png_path)
        plt.figure(figsize=(9, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
