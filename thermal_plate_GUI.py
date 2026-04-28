#!/usr/bin/env python3
"""
thermal_plate_sim_v15_2_gui.py

Cross-platform thermal plate simulator with segmented fin transfer, improved 3D viewer, and multi-worker optimization.
Run this file from the same folder as thermal_core.py.

Install dependencies:
    python -m pip install numpy matplotlib

Run:
    python thermal_plate_sim_v15_2_gui.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import queue
import sys
import threading
import traceback

# macOS: silence Tk deprecation noise on some Python/Tk combinations.
# This must be set before importing tkinter.
if sys.platform == "darwin":
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from thermal_core import (
    AIR_MOVEMENT_OPTIONS,
    ENVIRONMENT_OPTIONS,
    HOT_AIR_PATH_OPTIONS,
    MATERIALS,
    ORIENTATION_OPTIONS,
    SURFACE_OPTIONS,
    PlateConfig,
    Resistor,
    SimulationResult,
    SimulationSnapshot,
    display_from_key,
    estimate_passive_h,
    evenly_spaced_positions,
    fin_efficiency_for_spec,
    fin_temperature_at_height,
    format_time,
    heatsink_fin_specs,
    heatsink_fin_segment_specs,
    heatsink_geometry_summary,
    key_from_display,
    optimize_layout_deep,
    parse_float,
    parse_time_to_seconds,
    run_simulation,
    safe_time_name,
    save_temperature_grid_csv,
)


class ThermalPlateGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thermal Plate Simulator v15.2")

        # Cross-platform initial window sizing.
        # macOS laptops and smaller Linux/Windows screens can be shorter than
        # the old fixed 1320x840 window.
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = max(920, min(1380, screen_w - 70))
        win_h = max(620, min(920, screen_h - 90))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(min(860, win_w), min(600, win_h))

        self.resistors: List[Resistor] = [Resistor("R1", 50.0, 0.0, 0.0, 50.0, 20.0)]
        self.result: Optional[SimulationResult] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.msg_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self._ignore_slider_callback = False
        self._drawing_snapshot = False
        self._current_snapshot_idx: Optional[int] = None
        self._preview_after_id = None
        self._live_preview_ready = False
        self._optimization_running = False
        self.vmin: Optional[float] = None
        self.vmax: Optional[float] = None
        self._calibrated_h_value: Optional[float] = None
        self.e_last_power_each: Optional[float] = None

        self._create_vars()
        self._build_ui()
        self._refresh_resistor_tree()
        self._update_material_fields()
        self._update_h_label()
        self._setup_traces()
        self._live_preview_ready = True
        self._draw_layout_preview()
        self._install_cross_platform_shortcuts()
        self._install_native_menu()
        self.after(100, self._poll_queue)

    def _mousewheel_units(self, event) -> int:
        """Return scroll units for Windows/macOS/Linux Tk mouse wheel events."""
        # Linux/X11 usually sends Button-4/Button-5 instead of MouseWheel.
        num = getattr(event, "num", None)
        if num == 4:
            return -3
        if num == 5:
            return 3

        delta = getattr(event, "delta", 0)
        if delta == 0:
            return 0

        if sys.platform == "darwin":
            # macOS trackpads often send small deltas. Preserve direction and
            # avoid int(delta / 120) becoming zero.
            return -1 if delta > 0 else 1

        # Windows usually sends +/-120 per wheel notch.
        units = int(-delta / 120)
        if units == 0:
            units = -1 if delta > 0 else 1
        return units

    def _install_cross_platform_shortcuts(self):
        """Keyboard shortcuts with Ctrl on Windows/Linux and Command on macOS."""
        mod = "Command" if sys.platform == "darwin" else "Control"
        self.bind_all(f"<{mod}-s>", lambda e: (self._save_config(), "break"))
        self.bind_all(f"<{mod}-o>", lambda e: (self._load_config(), "break"))
        self.bind_all(f"<{mod}-r>", lambda e: (self._start_simulation(), "break"))
        self.bind_all("<Escape>", lambda e: (self._cancel_simulation(), "break"))

    def _install_native_menu(self):
        """Small native menu bar. Especially useful on macOS."""
        try:
            menubar = tk.Menu(self)

            file_menu = tk.Menu(menubar, tearoff=False)
            file_menu.add_command(label="Run Simulation", command=self._start_simulation)
            file_menu.add_separator()
            file_menu.add_command(label="Save Config...", command=self._save_config)
            file_menu.add_command(label="Load Config...", command=self._load_config)
            file_menu.add_separator()
            file_menu.add_command(label="Export Heatmap...", command=self._export_image)
            file_menu.add_command(label="Export CSV...", command=self._export_csv)
            file_menu.add_separator()
            file_menu.add_command(label="Quit", command=self.destroy)
            menubar.add_cascade(label="File", menu=file_menu)

            help_menu = tk.Menu(menubar, tearoff=False)
            help_menu.add_command(
                label="About",
                command=lambda: messagebox.showinfo(
                    "Thermal Plate Simulator v15.2",
                    "Thermal Plate Simulator v15.2\nCross-platform GUI for Windows, macOS, and Linux."
                )
            )
            menubar.add_cascade(label="Help", menu=help_menu)

            self.config(menu=menubar)
        except Exception:
            # Menus are convenience only; never block the simulator if a Tk
            # build handles menus differently.
            pass

    # ----------------------------- variables -----------------------------
    def _create_vars(self):
        self.plate_width_x_var = tk.StringVar(value="25")
        self.plate_height_y_var = tk.StringVar(value="25")
        self.plate_thickness_var = tk.StringVar(value="5")
        self.material_var = tk.StringVar(value="aluminium")
        self.k_var = tk.StringVar(value="205")
        self.rho_var = tk.StringVar(value="2700")
        self.cp_var = tk.StringVar(value="900")

        self.ambient_var = tk.StringVar(value="25")
        self.h_var = tk.StringVar(value="7")
        self.grid_var = tk.StringVar(value="5")
        self.initial_temp_var = tk.StringVar(value="25")
        self.max_time_var = tk.StringVar(value="1h")
        self.snapshot_every_var = tk.StringVar(value="1m")
        self.include_steady_var = tk.BooleanVar(value=True)
        self.fixed_scale_var = tk.BooleanVar(value=True)
        self.show_fin_overlay_2d_var = tk.BooleanVar(value=True)

        self.max_plate_temp_var = tk.StringVar(value="90")
        self.max_res_case_temp_var = tk.StringVar(value="120")
        self.max_res_element_temp_var = tk.StringVar(value="200")
        self.extra_cw_var = tk.StringVar(value="0.5")
        self.element_cw_var = tk.StringVar(value="0.6")
        self.cooling_margin_var = tk.StringVar(value="25")

        self.advanced_cooling_var = tk.BooleanVar(value=False)
        self.orientation_var = tk.StringVar(value=ORIENTATION_OPTIONS["vertical"])
        self.environment_var = tk.StringVar(value=ENVIRONMENT_OPTIONS["open_air"])
        self.clearance_var = tk.StringVar(value="20")
        self.surface_var = tk.StringVar(value=SURFACE_OPTIONS["bare_metal"])
        self.air_movement_var = tk.StringVar(value=AIR_MOVEMENT_OPTIONS["still_air"])
        self.hot_air_path_var = tk.StringVar(value=HOT_AIR_PATH_OPTIONS["free_rise"])
        self.blockage_var = tk.StringVar(value="0")
        self.h_label_var = tk.StringVar(value="Manual h = 7 W/m²K")
        self.cooling_notes = "manual h"

        self.heatsink_enabled_var = tk.BooleanVar(value=False)
        self.heatsink_mode_var = tk.StringVar(value="Geometry builder")
        self.heatsink_area_var = tk.StringVar(value="0")
        self.heatsink_eff_var = tk.StringVar(value="70")
        self.heatsink_hmul_var = tk.StringVar(value="1.0")
        self.heatsink_fin_orientation_var = tk.StringVar(value="Fins run along Y, spread across X")
        self.heatsink_fin_count_var = tk.StringVar(value="8")
        self.heatsink_fin_thickness_var = tk.StringVar(value="same")
        self.heatsink_fin_height_var = tk.StringVar(value="30")
        self.heatsink_fin_run_length_var = tk.StringVar(value="full")
        self.heatsink_fin_positions_var = tk.StringVar(value="even")
        self.heatsink_fin_heights_var = tk.StringVar(value="same")
        self.heatsink_fin_segments_var = tk.StringVar(value="auto")
        self.heatsink_label_var = tk.StringVar(value="No heatsink / extra fins")

        self.r_name_var = tk.StringVar(value="R1")
        self.r_power_var = tk.StringVar(value="50")
        self.r_x_var = tk.StringVar(value="0")
        self.r_y_var = tk.StringVar(value="0")
        self.r_len_var = tk.StringVar(value="50")
        self.r_wid_var = tk.StringVar(value="20")

        self.bank_count_var = tk.StringVar(value="4")
        self.bank_power_var = tk.StringVar(value="50")
        self.bank_len_var = tk.StringVar(value="50")
        self.bank_wid_var = tk.StringVar(value="20")
        self.bank_margin_var = tk.StringVar(value="1")
        self.optimizer_depth_var = tk.StringVar(value="Deep")
        self.optimizer_grid_var = tk.StringVar(value="12")
        self.optimizer_workers_var = tk.StringVar(value="0")
        self.resistor_side_var = tk.StringVar(value="Front / flat side")
        self.viewer_3d_quality_var = tk.StringVar(value="Canvas detailed")

        self.e_supply_v_var = tk.StringVar(value="28")
        self.e_res_ohm_var = tk.StringVar(value="1")
        self.e_count_var = tk.StringVar(value="4")
        self.e_connection_var = tk.StringVar(value="Series")
        self.e_result_var = tk.StringVar(value="Enter values and click Calculate.")

        self.size_power_var = tk.StringVar(value="50")
        self.size_target_var = tk.StringVar(value="90")
        self.size_ambient_var = tk.StringVar(value="25")
        self.size_h_var = tk.StringVar(value="5")
        self.size_result_var = tk.StringVar(value="")

        self.cal_power_var = tk.StringVar(value="50")
        self.cal_area_var = tk.StringVar(value="1250")
        self.cal_ambient_var = tk.StringVar(value="25")
        self.cal_measured_var = tk.StringVar(value="80")
        self.cal_result_var = tk.StringVar(value="")

    # ------------------------------- layout -------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(main, padding=8)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.tabs = ttk.Notebook(left)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.tab_basic = self._scroll_tab("Basic")
        self.tab_res = self._scroll_tab("Resistors")
        self.tab_tools = self._scroll_tab("Tools")
        self.tab_results = self._scroll_tab("Results")
        self.tab_help = self._scroll_tab("Help")

        self._build_basic_tab(self.tab_basic)
        self._build_resistor_tab(self.tab_res)
        self._build_tools_tab(self.tab_tools)
        self._build_results_tab(self.tab_results)
        self._build_help_tab(self.tab_help)

        right = ttk.Frame(main, padding=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self._build_plot(right)
        main.add(left, weight=2)
        main.add(right, weight=3)
        # Keep left panel practical on small screens while allowing expansion.
        self.after(10, lambda: main.sashpos(0, max(360, int(self.winfo_width() * 0.36))))

    def _scroll_tab(self, title: str) -> ttk.Frame:
        outer = ttk.Frame(self.tabs)
        self.tabs.add(outer, text=title)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        sb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=sb.set)

        inner = ttk.Frame(canvas, padding=8)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def fit_width(event=None):
            canvas.itemconfigure(win, width=canvas.winfo_width())

        def on_wheel(event):
            units = self._mousewheel_units(event)
            if units:
                canvas.yview_scroll(units, "units")
            return "break"

        def bind_wheel(event=None):
            # Use bind_all while hovering so the wheel still works when the
            # pointer is over entries/buttons inside the tab.
            canvas.bind_all("<MouseWheel>", on_wheel)
            canvas.bind_all("<Button-4>", on_wheel)
            canvas.bind_all("<Button-5>", on_wheel)

        def unbind_wheel(event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", fit_width)
        canvas.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)

        inner.columnconfigure(0, weight=1)
        return inner

    def _build_scrollable_dialog(self, win, width=620, height=650):
        """Create a resize-friendly vertically scrollable frame inside a dialog."""
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        win.geometry(f"{width}x{height}")
        win.minsize(520, 420)

        outer = ttk.Frame(win)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        sb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=sb.set)

        body = ttk.Frame(canvas, padding=12)
        body_win = canvas.create_window((0, 0), window=body, anchor="nw")

        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(body_win, width=canvas.winfo_width()))

        def on_wheel(event):
            units = self._mousewheel_units(event)
            if units:
                canvas.yview_scroll(units, "units")
            return "break"

        def bind_wheel(event=None):
            canvas.bind_all("<MouseWheel>", on_wheel)
            canvas.bind_all("<Button-4>", on_wheel)
            canvas.bind_all("<Button-5>", on_wheel)

        def unbind_wheel(event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        for widget in (canvas, body):
            widget.bind("<Enter>", bind_wheel)
            widget.bind("<Leave>", unbind_wheel)
        body.columnconfigure(1, weight=1)
        return body

    def _entry_row(self, parent, row, label, var, width=14):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)
        return e

    def _note(self, parent, row, text):
        ttk.Label(parent, text=text, foreground="#555", wraplength=360, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_basic_tab(self, parent):
        row = 0
        plate = ttk.LabelFrame(parent, text="Plate", padding=8)
        plate.grid(row=row, column=0, sticky="ew", pady=(0,8)); row += 1
        self._entry_row(plate, 0, "Width / X cm", self.plate_width_x_var)
        self._entry_row(plate, 1, "Height / Y cm", self.plate_height_y_var)
        self._entry_row(plate, 2, "Thickness mm", self.plate_thickness_var)
        ttk.Label(plate, text="Material").grid(row=3, column=0, sticky="w", pady=2)
        mat = ttk.Combobox(plate, textvariable=self.material_var, values=list(MATERIALS.keys()), state="readonly")
        mat.grid(row=3, column=1, sticky="ew", pady=2)
        mat.bind("<<ComboboxSelected>>", self._material_changed)
        self.k_entry = self._entry_row(plate, 4, "k W/mK", self.k_var)
        self.rho_entry = self._entry_row(plate, 5, "Density kg/m³", self.rho_var)
        self.cp_entry = self._entry_row(plate, 6, "Heat cap J/kgK", self.cp_var)
        self._note(plate, 7, "Template metals lock material fields. Choose custom to edit.")

        sim = ttk.LabelFrame(parent, text="Simulation", padding=8)
        sim.grid(row=row, column=0, sticky="ew", pady=(0,8)); row += 1
        self._entry_row(sim, 0, "Ambient °C", self.ambient_var)
        self._entry_row(sim, 1, "Manual h W/m²K", self.h_var)
        self._entry_row(sim, 2, "Grid mm", self.grid_var)
        self._entry_row(sim, 3, "Initial °C", self.initial_temp_var)
        self._entry_row(sim, 4, "Max time", self.max_time_var)
        self._entry_row(sim, 5, "Snapshot every", self.snapshot_every_var)
        ttk.Checkbutton(sim, text="Include steady-state final", variable=self.include_steady_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4,0))
        ttk.Checkbutton(sim, text="Use advanced cooling estimate", variable=self.advanced_cooling_var, command=self._update_h_label).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4,0))
        ttk.Button(sim, text="Advanced cooling...", command=self._advanced_dialog).grid(row=8, column=0, sticky="ew", pady=(6,0))
        ttk.Label(sim, textvariable=self.h_label_var, foreground="#555", wraplength=220).grid(row=8, column=1, sticky="w", padx=(8,0), pady=(6,0))
        ttk.Button(sim, text="Heatsink / fins...", command=self._heatsink_dialog).grid(row=9, column=0, sticky="ew", pady=(6,0))
        ttk.Label(sim, textvariable=self.heatsink_label_var, foreground="#555", wraplength=220).grid(row=9, column=1, sticky="w", padx=(8,0), pady=(6,0))
        self._note(sim, 10, "Use h=5 for conservative passive open-air checks. Heatsink support is optional and modeled as extra effective area.")

        safety = ttk.LabelFrame(parent, text="Optional pass/fail limits", padding=8)
        safety.grid(row=row, column=0, sticky="ew", pady=(0,8)); row += 1
        self._entry_row(safety, 0, "Max plate °C", self.max_plate_temp_var)
        self._entry_row(safety, 1, "Max resistor case °C", self.max_res_case_temp_var)
        self._entry_row(safety, 2, "Max element °C", self.max_res_element_temp_var)
        self._entry_row(safety, 3, "Case→plate °C/W", self.extra_cw_var)
        self._entry_row(safety, 4, "Element→case °C/W", self.element_cw_var)
        self._entry_row(safety, 5, "Cooling margin %", self.cooling_margin_var)
        self._note(safety, 6, "Optional. Use datasheet °C/W values when available. Otherwise defaults are placeholders.")

        run = ttk.LabelFrame(parent, text="Run", padding=8)
        run.grid(row=row, column=0, sticky="ew", pady=(0,8)); row += 1
        self.run_button = ttk.Button(run, text="Run simulation", command=self._start_simulation)
        self.run_button.grid(row=0, column=0, sticky="ew", pady=2)
        self.cancel_button = ttk.Button(run, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(4,0), pady=2)
        ttk.Button(run, text="Save config", command=self._save_config).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(run, text="Load config", command=self._load_config).grid(row=1, column=1, sticky="ew", padx=(4,0), pady=2)
        ttk.Checkbutton(run, text="Fixed heatmap color scale", variable=self.fixed_scale_var, command=self._redraw_current).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))
        ttk.Checkbutton(run, text="Show fins in 2D heatmap", variable=self.show_fin_overlay_2d_var, command=self._redraw_current).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4,0))

    def _build_resistor_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="Resistors", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0,8))
        cols = ("name","power","x","y","len","wid")
        self.res_tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c, h, w in [("name","Name",58),("power","W",50),("x","x cm",55),("y","y cm",55),("len","L mm",58),("wid","W mm",58)]:
            self.res_tree.heading(c, text=h); self.res_tree.column(c, width=w, anchor="center")
        self.res_tree.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.res_tree.bind("<<TreeviewSelect>>", self._res_selected)
        fields = [("Name",self.r_name_var),("Power W",self.r_power_var),("x cm",self.r_x_var),("y cm",self.r_y_var),("Len mm",self.r_len_var),("Wid mm",self.r_wid_var)]
        for i,(lab,var) in enumerate(fields):
            ttk.Label(frame, text=lab).grid(row=1+i//2, column=(i%2)*2, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=var, width=10).grid(row=1+i//2, column=(i%2)*2+1, sticky="ew", pady=2)
        ttk.Label(frame, text="3D display side").grid(row=4, column=0, sticky="w", pady=(8,2))
        ttk.Combobox(frame, textvariable=self.resistor_side_var, values=["Front / flat side", "Back / fin side", "Both sides"], state="readonly", width=18).grid(row=4, column=1, columnspan=3, sticky="ew", pady=(8,2))
        btn = ttk.Frame(frame); btn.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8,0))
        ttk.Button(btn, text="Add", command=self._add_res).pack(side="left", padx=(0,4))
        ttk.Button(btn, text="Update", command=self._update_res).pack(side="left", padx=(0,4))
        ttk.Button(btn, text="Delete", command=self._delete_res).pack(side="left")

        bank = ttk.LabelFrame(parent, text="Create / optimize bank", padding=8)
        bank.grid(row=1, column=0, sticky="ew", pady=(0,8))
        for i,(lab,var) in enumerate([("Count",self.bank_count_var),("W each",self.bank_power_var),("Len mm",self.bank_len_var),("Wid mm",self.bank_wid_var),("Margin cm",self.bank_margin_var)]):
            ttk.Label(bank, text=lab).grid(row=i//2, column=(i%2)*2, sticky="w", pady=2)
            ttk.Entry(bank, textvariable=var, width=9).grid(row=i//2, column=(i%2)*2+1, sticky="ew", pady=2)
        ttk.Label(bank, text="Depth").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Combobox(bank, textvariable=self.optimizer_depth_var, values=["Balanced","Deep","Extreme"], state="readonly", width=10).grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(bank, text="Opt grid mm").grid(row=3, column=2, sticky="w", pady=2)
        ttk.Entry(bank, textvariable=self.optimizer_grid_var, width=9).grid(row=3, column=3, sticky="ew", pady=2)
        ttk.Label(bank, text="Workers").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(bank, textvariable=self.optimizer_workers_var, width=9).grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Button(bank, text="Create even bank", command=self._create_bank).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8,0))
        ttk.Button(bank, text="Deep optimize", command=self._start_optimizer).grid(row=5, column=2, columnspan=2, sticky="ew", padx=(4,0), pady=(8,0))
        self._note(bank, 6, "Workers: 0 = auto. Multi-worker is used during optimizer candidate heat-solves. Use Create even bank first, then Deep optimize if you want the layout improved.")

    def _build_tools_tab(self, parent):
        e = ttk.LabelFrame(parent, text="Electrical power calculator", padding=8)
        e.grid(row=0, column=0, sticky="ew", pady=(0,8))
        self._entry_row(e,0,"Supply V",self.e_supply_v_var)
        self._entry_row(e,1,"Each resistor Ω",self.e_res_ohm_var)
        self._entry_row(e,2,"Count",self.e_count_var)
        ttk.Label(e,text="Connection").grid(row=3,column=0,sticky="w",pady=2)
        ttk.Combobox(e,textvariable=self.e_connection_var,values=["Series","Parallel"],state="readonly").grid(row=3,column=1,sticky="ew",pady=2)
        ttk.Button(e,text="Calculate",command=self._calc_electrical).grid(row=4,column=0,sticky="ew",pady=(6,0))
        ttk.Button(e,text="Apply W each",command=self._apply_electrical).grid(row=4,column=1,sticky="ew",padx=(4,0),pady=(6,0))
        ttk.Label(e,textvariable=self.e_result_var,wraplength=360,justify="left").grid(row=5,column=0,columnspan=2,sticky="w",pady=(8,0))

        s = ttk.LabelFrame(parent, text="Required plate area estimate", padding=8)
        s.grid(row=1,column=0,sticky="ew",pady=(0,8))
        self._entry_row(s,0,"Power W",self.size_power_var)
        self._entry_row(s,1,"Target plate °C",self.size_target_var)
        self._entry_row(s,2,"Ambient °C",self.size_ambient_var)
        self._entry_row(s,3,"h W/m²K",self.size_h_var)
        ttk.Button(s,text="Estimate",command=self._calc_size).grid(row=4,column=0,sticky="ew",pady=(6,0))
        ttk.Label(s,textvariable=self.size_result_var,wraplength=360,justify="left").grid(row=5,column=0,columnspan=2,sticky="w",pady=(8,0))

        c = ttk.LabelFrame(parent, text="Estimate h from real test", padding=8)
        c.grid(row=2,column=0,sticky="ew",pady=(0,8))
        self._entry_row(c,0,"Test power W",self.cal_power_var)
        self._entry_row(c,1,"Area both sides cm²",self.cal_area_var)
        self._entry_row(c,2,"Ambient °C",self.cal_ambient_var)
        self._entry_row(c,3,"Measured avg plate °C",self.cal_measured_var)
        ttk.Button(c,text="Estimate h",command=self._calc_h_cal).grid(row=4,column=0,sticky="ew",pady=(6,0))
        ttk.Button(c,text="Use this h",command=self._use_cal_h).grid(row=4,column=1,sticky="ew",padx=(4,0),pady=(6,0))
        ttk.Label(c,textvariable=self.cal_result_var,wraplength=360,justify="left").grid(row=5,column=0,columnspan=2,sticky="w",pady=(8,0))

    def _build_results_tab(self, parent):
        actions = ttk.Frame(parent); actions.grid(row=0,column=0,sticky="ew",pady=(0,8))
        ttk.Button(actions,text="Export heatmap PNG",command=self._export_image).pack(side="left",padx=(0,4))
        ttk.Button(actions,text="Export grid CSV",command=self._export_csv).pack(side="left")
        self.status_text = tk.Text(parent,width=48,height=30,wrap="word")
        self.status_text.grid(row=1,column=0,sticky="nsew")
        sb=ttk.Scrollbar(parent,orient="vertical",command=self.status_text.yview); sb.grid(row=1,column=1,sticky="ns")
        self.status_text.configure(yscrollcommand=sb.set)
        parent.rowconfigure(1,weight=1); parent.columnconfigure(0,weight=1)
        self._set_status("Ready. Use Basic + Resistors, then run simulation.")

    def _build_help_tab(self, parent):
        t = tk.Text(parent,width=48,height=30,wrap="word")
        t.grid(row=0,column=0,sticky="nsew")
        sb=ttk.Scrollbar(parent,orient="vertical",command=t.yview); sb.grid(row=0,column=1,sticky="ns")
        t.configure(yscrollcommand=sb.set)
        parent.rowconfigure(0,weight=1); parent.columnconfigure(0,weight=1)
        t.insert("1.0", """
Simple workflow
1. Basic tab: enter plate size, material, cooling, and time settings.
2. Resistors tab: add or create the resistor bank. The live preview updates without running the heat simulation.
3. Optional: use Tools for electrical power, rough required plate area, or h calibration.
4. Run simulation and read the Results tab.

Key values
Width/X is the horizontal size in the preview. Height/Y is vertical.
Thickness improves heat spreading and delays warm-up; surface area controls continuous passive cooling.
k W/mK is material heat spreading. Aluminium is much better than steel.
h W/m²K is passive cooling strength. Lower h means worse cooling. For conservative passive work, test with h=5.
Grid mm controls simulation detail. 5 mm is a good default. Smaller is slower.

Safety values
Extra resistor→plate °C/W estimates the resistor case being hotter than the simulated plate contact. Example: 50 W × 0.5 °C/W adds 25 °C.
Cooling margin % is only for the pass/fail interpretation.

Optimizer
Balanced is faster. Deep is the normal choice. Extreme searches harder and can take much longer.
Always run the full simulation after optimizing.

3D viewer
Open 3D viewer after a simulation to see the heatmapped plate, back-side fins, and resistor blocks. v12 adds transparent/exploded viewing and resistor outlines, so backside resistors are much easier to see from awkward angles. The resistor side control is visual; the 2D thermal model is still through the same thin plate.

Fin layout and fin heat transfer
Use the heatsink dialog's Fin layout designer for individual fin positions and heights. Even fills the available width. Place fins near resistors clusters fins under/near heat sources along the fin spread axis. Thermal segments controls how finely each fin is split along its length for local heat-transfer calculation. Use auto unless you specifically need more detail. v14 also colors fins with a temperature gradient up their height using the fin equation, instead of coloring the whole fin uniformly.

Resistor body/element temperatures
The plate solver gives the local plate footprint temperature. v14+ adds a resistor thermal stack: plate footprint → case/body → internal element. Enter Case→plate °C/W and Element→case °C/W from a datasheet if possible.

Performance
v15.2 replaces the Matplotlib 3D viewer with a custom Tk Canvas engineering renderer. It projects the plate, fins, and resistors directly to 2D polygons, which is much faster for this fixed-geometry simulator than mplot3d.
""".strip())
        t.configure(state="disabled")

    def _build_plot(self, parent):
        frame=ttk.Frame(parent); frame.grid(row=0,column=0,sticky="nsew"); frame.rowconfigure(0,weight=1); frame.columnconfigure(0,weight=1)
        self.fig=Figure(figsize=(8,6),dpi=100); self.ax=self.fig.add_subplot(111); self.colorbar=None
        self.canvas=FigureCanvasTkAgg(self.fig,master=frame); self.canvas.get_tk_widget().grid(row=0,column=0,sticky="nsew")
        tb=ttk.Frame(frame); tb.grid(row=1,column=0,sticky="ew"); self.toolbar=NavigationToolbar2Tk(self.canvas,tb); self.toolbar.update()
        slider=ttk.Frame(parent,padding=(0,8,0,0)); slider.grid(row=1,column=0,sticky="ew"); slider.columnconfigure(1,weight=1)
        ttk.Label(slider,text="Time").grid(row=0,column=0,sticky="w",padx=(0,6))
        self.time_slider=ttk.Scale(slider,from_=0,to=0,orient="horizontal",command=self._slider_changed); self.time_slider.grid(row=0,column=1,sticky="ew")
        self.snapshot_label_var=tk.StringVar(value="Live preview"); ttk.Label(slider,textvariable=self.snapshot_label_var,width=28).grid(row=0,column=2,sticky="e",padx=(8,0))
        nav=ttk.Frame(parent); nav.grid(row=2,column=0,sticky="ew")
        ttk.Button(nav,text="Previous",command=lambda:self._step_slider(-1)).pack(side="left")
        ttk.Button(nav,text="Next",command=lambda:self._step_slider(1)).pack(side="left",padx=(4,0))
        ttk.Button(nav,text="Show live layout",command=self._draw_layout_preview).pack(side="left",padx=(16,0))
        ttk.Button(nav,text="Open 3D viewer",command=self._open_3d_viewer).pack(side="left",padx=(4,0))

    # ------------------------------ behavior ------------------------------
    def _setup_traces(self):
        for var in [self.plate_width_x_var,self.plate_height_y_var,self.plate_thickness_var,self.k_var,self.rho_var,self.cp_var,self.ambient_var,self.h_var,self.grid_var,self.r_power_var,self.r_x_var,self.r_y_var,self.r_len_var,self.r_wid_var]:
            var.trace_add("write", lambda *a: self._schedule_preview())
        for var in [self.advanced_cooling_var,self.orientation_var,self.environment_var,self.clearance_var,self.surface_var,self.air_movement_var,self.hot_air_path_var,self.blockage_var]:
            var.trace_add("write", lambda *a: self._update_h_label())
        for var in [
            self.heatsink_enabled_var,self.heatsink_mode_var,self.heatsink_area_var,self.heatsink_eff_var,self.heatsink_hmul_var,
            self.heatsink_fin_orientation_var,self.heatsink_fin_count_var,self.heatsink_fin_thickness_var,
            self.heatsink_fin_height_var,self.heatsink_fin_run_length_var,self.heatsink_fin_positions_var,
            self.heatsink_fin_heights_var,self.heatsink_fin_segments_var
        ]:
            var.trace_add("write", lambda *a: (self._update_heatsink_label(), self._schedule_preview()))
        self.resistor_side_var.trace_add("write", lambda *a: self._schedule_preview())

    def _schedule_preview(self):
        if not self._live_preview_ready: return
        if self._preview_after_id:
            try: self.after_cancel(self._preview_after_id)
            except Exception: pass
        self._preview_after_id = self.after(120, self._draw_layout_preview)

    def _material_changed(self, event=None):
        name=self.material_var.get()
        if name in MATERIALS and name != "custom":
            m=MATERIALS[name]; self.k_var.set(str(m["k"])); self.rho_var.set(str(m["rho"])); self.cp_var.set(str(m["cp"]))
        self._update_material_fields(); self._schedule_preview()

    def _update_material_fields(self):
        st="normal" if self.material_var.get()=="custom" else "disabled"
        for e in [getattr(self,"k_entry",None),getattr(self,"rho_entry",None),getattr(self,"cp_entry",None)]:
            if e: e.configure(state=st)

    def _update_h_label(self):
        try:
            if self.advanced_cooling_var.get():
                h, notes = estimate_passive_h(
                    key_from_display(ORIENTATION_OPTIONS,self.orientation_var.get()),
                    key_from_display(ENVIRONMENT_OPTIONS,self.environment_var.get()),
                    parse_float(self.clearance_var.get(),"Clearance",0.0),
                    key_from_display(SURFACE_OPTIONS,self.surface_var.get()),
                    key_from_display(AIR_MOVEMENT_OPTIONS,self.air_movement_var.get()),
                    key_from_display(HOT_AIR_PATH_OPTIONS,self.hot_air_path_var.get()),
                    parse_float(self.blockage_var.get(),"Blockage",0.0),
                )
                self.h_var.set(f"{h:.2f}"); self.cooling_notes=notes; self.h_label_var.set(f"Advanced h ≈ {h:.2f} W/m²K")
            else:
                self.cooling_notes="manual h"; self.h_label_var.set(f"Manual h = {self.h_var.get()} W/m²K")
        except Exception as e:
            self.h_label_var.set(f"Cooling estimate error: {e}")

    def _advanced_dialog(self):
        win=tk.Toplevel(self); win.title("Advanced cooling"); win.transient(self); win.grab_set()
        f = self._build_scrollable_dialog(win, width=620, height=560)
        ttk.Checkbutton(f,text="Use advanced estimate",variable=self.advanced_cooling_var,command=self._update_h_label).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,8))
        def combo(row,label,var,opts):
            ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3)
            cb=ttk.Combobox(f,textvariable=var,values=list(opts.values()),state="readonly",width=34); cb.grid(row=row,column=1,sticky="ew",pady=3)
            cb.bind("<<ComboboxSelected>>",lambda e:self._update_h_label())
        combo(1,"Orientation",self.orientation_var,ORIENTATION_OPTIONS)
        combo(2,"Environment",self.environment_var,ENVIRONMENT_OPTIONS)
        self._entry_row(f,3,"Wall gap cm",self.clearance_var)
        combo(4,"Surface",self.surface_var,SURFACE_OPTIONS)
        combo(5,"Air movement",self.air_movement_var,AIR_MOVEMENT_OPTIONS)
        combo(6,"Hot air path",self.hot_air_path_var,HOT_AIR_PATH_OPTIONS)
        self._entry_row(f,7,"Blockage %",self.blockage_var)
        ttk.Label(f,textvariable=self.h_label_var,foreground="#555",wraplength=420).grid(row=8,column=0,columnspan=2,sticky="w",pady=(10,0))
        ttk.Button(f,text="Close",command=win.destroy).grid(row=9,column=0,columnspan=2,sticky="ew",pady=(12,0))

    def _heatsink_orientation_key(self) -> str:
        return "run_x" if "along X" in self.heatsink_fin_orientation_var.get() else "run_y"

    def _float_or_zero_for_full_same(self, value: str, default: float = 0.0) -> float:
        raw = str(value or "").strip().lower()
        if raw in ("", "same", "full", "auto"):
            return default
        return parse_float(raw, "value", 0.0)

    def _int_or_zero_for_auto(self, value: str) -> int:
        raw = str(value or "").strip().lower()
        if raw in ("", "auto", "same", "full"):
            return 0
        return max(0, int(round(parse_float(raw, "segments", 0))))


    def _update_heatsink_label(self):
        try:
            if not self.heatsink_enabled_var.get():
                self.heatsink_label_var.set("No heatsink / extra fins")
                return

            if self.heatsink_mode_var.get().startswith("Geometry"):
                cfg = self._read_config_loose_for_heatsink()
                summary = heatsink_geometry_summary(cfg)
                if not summary["enabled"]:
                    self.heatsink_label_var.set("Geometry heatsink enabled, but no fins calculated")
                    return
                self.heatsink_label_var.set(
                    f"{summary['fin_count']} fins / {summary.get('thermal_segment_count', 0)} thermal segments, "
                    f"raw area {summary['raw_fin_area_cm2']:.0f} cm², "
                    f"effective extra ≈ {summary['effective_extra_area_cm2']:.0f} cm², "
                    f"avg fin η {summary['average_fin_efficiency_percent']:.0f}%"
                )
            else:
                area=parse_float(self.heatsink_area_var.get(),"Heatsink area",0.0)
                eff=parse_float(self.heatsink_eff_var.get(),"Heatsink efficiency",0.0)
                hmul=parse_float(self.heatsink_hmul_var.get(),"Heatsink h multiplier",0.0)
                effective=area*max(0,min(100,eff))/100*hmul
                self.heatsink_label_var.set(f"Simple extra effective area ≈ {effective:.0f} cm²")
        except Exception as e:
            self.heatsink_label_var.set(f"Heatsink error: {e}")

    def _read_config_loose_for_heatsink(self) -> PlateConfig:
        """Read just enough config for heatsink preview/summary without requiring a full run."""
        h = parse_float(self.h_var.get(), "h", 0.1)
        if self.advanced_cooling_var.get():
            try:
                h, _ = estimate_passive_h(
                    orientation=key_from_display(__import__('thermal_core').ORIENTATION_OPTIONS,self.orientation_var.get()),
                    environment=key_from_display(__import__('thermal_core').ENVIRONMENT_OPTIONS,self.environment_var.get()),
                    wall_clearance_cm=parse_float(self.clearance_var.get(),"Wall clearance",0.0),
                    surface_finish=key_from_display(__import__('thermal_core').SURFACE_OPTIONS,self.surface_var.get()),
                    air_movement=key_from_display(__import__('thermal_core').AIR_MOVEMENT_OPTIONS,self.air_movement_var.get()),
                    hot_air_path=key_from_display(__import__('thermal_core').HOT_AIR_PATH_OPTIONS,self.hot_air_path_var.get()),
                    blockage_percent=parse_float(self.blockage_var.get(),"Blockage",0.0),
                )
            except Exception:
                pass

        fin_t = self._float_or_zero_for_full_same(self.heatsink_fin_thickness_var.get(), 0.0)
        run_len = self._float_or_zero_for_full_same(self.heatsink_fin_run_length_var.get(), 0.0)

        return PlateConfig(
            plate_length_cm=parse_float(self.plate_width_x_var.get(),"Plate width",0.1),
            plate_width_cm=parse_float(self.plate_height_y_var.get(),"Plate height",0.1),
            plate_thickness_mm=parse_float(self.plate_thickness_var.get(),"Thickness",0.1),
            material_name=self.material_var.get(),
            thermal_conductivity_w_mk=parse_float(self.k_var.get(),"k",0.1),
            density_kg_m3=parse_float(self.rho_var.get(),"Density",1.0),
            heat_capacity_j_kgk=parse_float(self.cp_var.get(),"Heat capacity",1.0),
            ambient_c=parse_float(self.ambient_var.get(),"Ambient"),
            convection_h_w_m2k=h,
            grid_mm=max(1.0, self._float_or_zero_for_full_same(self.grid_var.get(), 5.0)),
            resistors=list(self.resistors) if self.resistors else [Resistor("R1", 0, 0, 0, 10, 10)],
            initial_plate_temp_c=parse_float(self.initial_temp_var.get(),"Initial"),
            max_time_s=60.0,
            snapshot_every_s=60.0,
            include_steady_state=False,
            resistor_case_to_plate_cw=self._f(self.extra_cw_var.get(), 0.5),
            resistor_element_to_case_cw=self._f(self.element_cw_var.get(), 0.6),
            heatsink_enabled=bool(self.heatsink_enabled_var.get()),
            heatsink_geometry_enabled=self.heatsink_mode_var.get().startswith("Geometry"),
            heatsink_fin_orientation=self._heatsink_orientation_key(),
            heatsink_fin_count=int(parse_float(self.heatsink_fin_count_var.get(),"Fin count",0)),
            heatsink_fin_thickness_mm=fin_t,
            heatsink_fin_default_height_mm=parse_float(self.heatsink_fin_height_var.get(),"Fin height",0.1),
            heatsink_fin_run_length_cm=run_len,
            heatsink_fin_positions_cm=self.heatsink_fin_positions_var.get(),
            heatsink_fin_heights_mm=self.heatsink_fin_heights_var.get(),
            heatsink_fin_segments=self._int_or_zero_for_auto(self.heatsink_fin_segments_var.get()),
            heatsink_extra_area_cm2=parse_float(self.heatsink_area_var.get(),"Heatsink extra area",0.0),
            heatsink_efficiency_percent=parse_float(self.heatsink_eff_var.get(),"Heatsink efficiency",0.0),
            heatsink_h_multiplier=parse_float(self.heatsink_hmul_var.get(),"Heatsink h multiplier",0.0),
        )

    def _heatsink_dialog(self):
        win=tk.Toplevel(self); win.title("Heatsink builder"); win.transient(self); win.grab_set()
        f = self._build_scrollable_dialog(win, width=720, height=680)
        ttk.Checkbutton(f,text="Enable heatsink / fins",variable=self.heatsink_enabled_var,command=self._update_heatsink_label).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,8))

        ttk.Label(f,text="Mode").grid(row=1,column=0,sticky="w",pady=3)
        ttk.Combobox(f,textvariable=self.heatsink_mode_var,values=["Geometry builder","Simple extra area"],state="readonly",width=30).grid(row=1,column=1,sticky="ew",pady=3)

        geom=ttk.LabelFrame(f,text="Geometry builder: fins on back, resistors on flat/front side",padding=8)
        geom.grid(row=2,column=0,columnspan=2,sticky="ew",pady=(8,8)); geom.columnconfigure(1,weight=1)
        ttk.Label(geom,text="Orientation").grid(row=0,column=0,sticky="w",pady=2)
        ttk.Combobox(
            geom,
            textvariable=self.heatsink_fin_orientation_var,
            values=["Fins run along Y, spread across X","Fins run along X, spread across Y"],
            state="readonly",
            width=34
        ).grid(row=0,column=1,sticky="ew",pady=2)
        self._entry_row(geom,1,"Number of fins",self.heatsink_fin_count_var)
        self._entry_row(geom,2,"Fin thickness mm",self.heatsink_fin_thickness_var)
        self._entry_row(geom,3,"Default fin height mm",self.heatsink_fin_height_var)
        self._entry_row(geom,4,"Fin run length cm",self.heatsink_fin_run_length_var)
        self._entry_row(geom,5,"Fin positions cm",self.heatsink_fin_positions_var)
        self._entry_row(geom,6,"Individual heights mm",self.heatsink_fin_heights_var)
        self._entry_row(geom,7,"Thermal segments",self.heatsink_fin_segments_var)
        ttk.Label(
            geom,
            text="Use 'same' for fin thickness to match the base plate. Use 'full' for run length to span the plate. "
                 "Positions: 'even' or comma-separated centers. Heights: 'same' or comma-separated values. Thermal segments: use 'auto' unless you want to force a specific segment count per fin.",
            foreground="#555",wraplength=420,justify="left"
        ).grid(row=8,column=0,columnspan=2,sticky="w",pady=(8,0))
        fin_btns=ttk.Frame(geom); fin_btns.grid(row=9,column=0,columnspan=2,sticky="ew",pady=(8,0))
        ttk.Button(fin_btns,text="Fin layout designer...",command=self._open_fin_layout_designer).pack(side="left",padx=(0,4))
        ttk.Button(fin_btns,text="Even fins",command=self._set_fins_even).pack(side="left",padx=(0,4))
        ttk.Button(fin_btns,text="Place fins near resistors",command=self._set_fins_near_resistors).pack(side="left")

        simple=ttk.LabelFrame(f,text="Simple area fallback",padding=8)
        simple.grid(row=3,column=0,columnspan=2,sticky="ew",pady=(0,8)); simple.columnconfigure(1,weight=1)
        self._entry_row(simple,0,"Extra exposed area cm²",self.heatsink_area_var)
        self._entry_row(simple,1,"Manual efficiency %",self.heatsink_eff_var)
        self._entry_row(simple,2,"h multiplier",self.heatsink_hmul_var)

        ttk.Label(f,textvariable=self.heatsink_label_var,foreground="#555",wraplength=460).grid(row=4,column=0,columnspan=2,sticky="w",pady=(10,0))
        ttk.Label(
            f,
            text="Geometry mode calculates fin area and fin efficiency from the dimensions. The remaining assumption is still the airflow/cooling h value, because passive air cannot be known exactly without measurement or CFD.",
            foreground="#555",wraplength=460,justify="left"
        ).grid(row=5,column=0,columnspan=2,sticky="w",pady=(8,0))
        ttk.Button(f,text="Close",command=win.destroy).grid(row=6,column=0,columnspan=2,sticky="ew",pady=(12,0))
        self._update_heatsink_label()

    # ----------------------------- config/read -----------------------------
    # ----------------------------- config/read -----------------------------
    def _read_config(self) -> PlateConfig:
        cfg=PlateConfig(
            plate_length_cm=parse_float(self.plate_width_x_var.get(),"Plate width",0.1),
            plate_width_cm=parse_float(self.plate_height_y_var.get(),"Plate height",0.1),
            plate_thickness_mm=parse_float(self.plate_thickness_var.get(),"Thickness",0.1),
            material_name=self.material_var.get(),
            thermal_conductivity_w_mk=parse_float(self.k_var.get(),"k",0.1),
            density_kg_m3=parse_float(self.rho_var.get(),"Density",1.0),
            heat_capacity_j_kgk=parse_float(self.cp_var.get(),"Heat capacity",1.0),
            ambient_c=parse_float(self.ambient_var.get(),"Ambient"),
            convection_h_w_m2k=parse_float(self.h_var.get(),"h",0.1),
            grid_mm=parse_float(self.grid_var.get(),"Grid",0.5),
            resistors=list(self.resistors),
            initial_plate_temp_c=parse_float(self.initial_temp_var.get(),"Initial"),
            max_time_s=parse_time_to_seconds(self.max_time_var.get(),"Max time"),
            snapshot_every_s=parse_time_to_seconds(self.snapshot_every_var.get(),"Snapshot every"),
            include_steady_state=bool(self.include_steady_var.get()),
            resistor_case_to_plate_cw=parse_float(self.extra_cw_var.get(), "Case→plate °C/W", 0.0),
            resistor_element_to_case_cw=parse_float(self.element_cw_var.get(), "Element→case °C/W", 0.0),
            advanced_cooling_enabled=bool(self.advanced_cooling_var.get()),
            orientation=key_from_display(ORIENTATION_OPTIONS,self.orientation_var.get()),
            environment=key_from_display(ENVIRONMENT_OPTIONS,self.environment_var.get()),
            wall_clearance_cm=parse_float(self.clearance_var.get(),"Wall clearance",0.0),
            surface_finish=key_from_display(SURFACE_OPTIONS,self.surface_var.get()),
            air_movement=key_from_display(AIR_MOVEMENT_OPTIONS,self.air_movement_var.get()),
            hot_air_path=key_from_display(HOT_AIR_PATH_OPTIONS,self.hot_air_path_var.get()),
            blockage_percent=parse_float(self.blockage_var.get(),"Blockage",0.0),
            cooling_notes=self.cooling_notes,
            heatsink_enabled=bool(self.heatsink_enabled_var.get()),
            heatsink_extra_area_cm2=parse_float(self.heatsink_area_var.get(),"Heatsink extra area",0.0),
            heatsink_efficiency_percent=parse_float(self.heatsink_eff_var.get(),"Heatsink efficiency",0.0),
            heatsink_h_multiplier=parse_float(self.heatsink_hmul_var.get(),"Heatsink h multiplier",0.0),
            heatsink_notes=self.heatsink_label_var.get(),
            heatsink_geometry_enabled=self.heatsink_mode_var.get().startswith("Geometry"),
            heatsink_fin_orientation=self._heatsink_orientation_key(),
            heatsink_fin_count=int(parse_float(self.heatsink_fin_count_var.get(),"Fin count",0)),
            heatsink_fin_thickness_mm=self._float_or_zero_for_full_same(self.heatsink_fin_thickness_var.get(),0.0),
            heatsink_fin_default_height_mm=parse_float(self.heatsink_fin_height_var.get(),"Fin height",0.1),
            heatsink_fin_run_length_cm=self._float_or_zero_for_full_same(self.heatsink_fin_run_length_var.get(),0.0),
            heatsink_fin_positions_cm=self.heatsink_fin_positions_var.get(),
            heatsink_fin_heights_mm=self.heatsink_fin_heights_var.get(),
            heatsink_fin_segments=self._int_or_zero_for_auto(self.heatsink_fin_segments_var.get()),
        )
        if cfg.snapshot_every_s > cfg.max_time_s: raise ValueError("Snapshot interval cannot be greater than max time.")
        if not cfg.resistors: raise ValueError("Add at least one resistor.")
        snaps=int(math.floor(cfg.max_time_s/cfg.snapshot_every_s))+1
        if snaps>600: raise ValueError(f"This would create about {snaps} snapshots. Increase snapshot interval or reduce max time.")
        nx=max(3,int(round((cfg.plate_length_cm/100)/(cfg.grid_mm/1000)))); ny=max(3,int(round((cfg.plate_width_cm/100)/(cfg.grid_mm/1000))))
        if nx*ny>220000: raise ValueError(f"Grid would be {nx}×{ny}={nx*ny} cells. Use larger Grid mm.")
        return cfg

    # ---------------------------- resistors ----------------------------
    def _refresh_resistor_tree(self):
        for i in self.res_tree.get_children(): self.res_tree.delete(i)
        for i,r in enumerate(self.resistors):
            self.res_tree.insert("","end",iid=str(i),values=(r.name,f"{r.power_w:g}",f"{r.center_x_cm:g}",f"{r.center_y_cm:g}",f"{r.length_mm:g}",f"{r.width_mm:g}"))
        self._schedule_preview()

    def _res_selected(self,event=None):
        sel=self.res_tree.selection()
        if not sel: return
        r=self.resistors[int(sel[0])]
        self.r_name_var.set(r.name); self.r_power_var.set(f"{r.power_w:g}"); self.r_x_var.set(f"{r.center_x_cm:g}"); self.r_y_var.set(f"{r.center_y_cm:g}"); self.r_len_var.set(f"{r.length_mm:g}"); self.r_wid_var.set(f"{r.width_mm:g}")

    def _read_res(self):
        return Resistor(self.r_name_var.get().strip() or f"R{len(self.resistors)+1}", parse_float(self.r_power_var.get(),"Power",0.0), parse_float(self.r_x_var.get(),"x"), parse_float(self.r_y_var.get(),"y"), parse_float(self.r_len_var.get(),"Length",0.1), parse_float(self.r_wid_var.get(),"Width",0.1))
    def _add_res(self):
        try: self.resistors.append(self._read_res()); self._refresh_resistor_tree()
        except Exception as e: messagebox.showerror("Invalid resistor",str(e))
    def _update_res(self):
        sel=self.res_tree.selection()
        if not sel: messagebox.showinfo("Update", "Select a resistor first."); return
        try: self.resistors[int(sel[0])]=self._read_res(); self._refresh_resistor_tree(); self.res_tree.selection_set(sel[0])
        except Exception as e: messagebox.showerror("Invalid resistor",str(e))
    def _delete_res(self):
        sel=self.res_tree.selection()
        if sel: del self.resistors[int(sel[0])]; self._refresh_resistor_tree()
    def _create_bank(self):
        try:
            count=int(parse_float(self.bank_count_var.get(),"Count",1)); power=parse_float(self.bank_power_var.get(),"Power",0.0)
            length=parse_float(self.bank_len_var.get(),"Length",0.1); width=parse_float(self.bank_wid_var.get(),"Width",0.1); margin=parse_float(self.bank_margin_var.get(),"Margin",0.0)
            px=parse_float(self.plate_width_x_var.get(),"Plate width",0.1); py=parse_float(self.plate_height_y_var.get(),"Plate height",0.1)
            pos=evenly_spaced_positions(count,px,py,length,width,margin)
            self.resistors=[Resistor(f"R{i+1}",power,x,y,length,width) for i,(x,y) in enumerate(pos)]
            self._refresh_resistor_tree(); self._set_status(f"Created even bank with {count} resistors.")
        except Exception as e: messagebox.showerror("Create bank failed",str(e))

    # ---------------------------- plotting ----------------------------
    def _draw_layout_preview(self):
        if not self._live_preview_ready or self._drawing_snapshot: return
        try: px=parse_float(self.plate_width_x_var.get(),"Plate width",0.1); py=parse_float(self.plate_height_y_var.get(),"Plate height",0.1)
        except Exception: return
        self._drawing_snapshot=True
        try:
            self.fig.clear(); self.ax=self.fig.add_subplot(111); self.colorbar=None
            self.ax.set_aspect("equal", adjustable="box"); self.ax.set_xlim(-px/2,px/2); self.ax.set_ylim(-py/2,py/2)
            self.ax.add_patch(Rectangle((-px/2,-py/2),px,py,fill=False,linewidth=2))

            # Blue dashed strips show heatsink fin footprints on the back side.
            try:
                if self.heatsink_enabled_var.get() and self.heatsink_mode_var.get().startswith("Geometry"):
                    cfg_preview = self._read_config_loose_for_heatsink()
                    for fin in heatsink_fin_specs(cfg_preview):
                        fx=fin["center_x_cm"]; fy=fin["center_y_cm"]; fw=fin["footprint_x_cm"]; fh=fin["footprint_y_cm"]
                        self.ax.add_patch(Rectangle((fx-fw/2,fy-fh/2),fw,fh,fill=False,linestyle="--",linewidth=1.4))
            except Exception:
                pass

            for r in self.resistors:
                l=r.length_mm/10; w=r.width_mm/10
                self.ax.add_patch(Rectangle((r.center_x_cm-l/2,r.center_y_cm-w/2),l,w,fill=True,alpha=0.25,linewidth=2))
                self.ax.text(r.center_x_cm,r.center_y_cm,f"{r.name}\n{r.power_w:g}W",ha="center",va="center",fontsize=9)
            self.ax.axhline(0,linewidth=.7,alpha=.3); self.ax.axvline(0,linewidth=.7,alpha=.3)
            self.ax.set_title("Live layout preview"); self.ax.set_xlabel("x / width, cm"); self.ax.set_ylabel("y / height, cm"); self.ax.grid(True,alpha=.25)
            self.fig.subplots_adjust(left=.08,right=.96,bottom=.10,top=.92); self.canvas.draw_idle(); self.snapshot_label_var.set("Live preview"); self._current_snapshot_idx=None
        finally: self._drawing_snapshot=False

    def _draw_fin_overlay_2d(self, ax, cfg):
        """Draw fin footprints on top of the 2D heatmap."""
        if not self.show_fin_overlay_2d_var.get():
            return
        try:
            if not getattr(cfg, "heatsink_geometry_enabled", False):
                return
            for fin in heatsink_fin_specs(cfg):
                fx = fin["center_x_cm"]
                fy = fin["center_y_cm"]
                fw = max(0.01, fin["footprint_x_cm"])
                fh = max(0.01, fin["footprint_y_cm"])
                ax.add_patch(
                    Rectangle(
                        (fx - fw / 2.0, fy - fh / 2.0),
                        fw,
                        fh,
                        fill=False,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.75,
                    )
                )
        except Exception:
            pass

    def _effective_3d_quality(self):
        """Return (fin_segment_stride, height_slices) for the 3D viewer.

        v15.1 changes the meaning:
        - Full / batched: draws every fin segment using batched collections.
        - Simplified: still available as an emergency mode for very old machines.
        """
        q = str(self.viewer_3d_quality_var.get()).lower()
        if "simpl" in q:
            return 3, 4
        return 1, 8

    def _box_faces_3d_translated(self, cx, cy, cz, sx, sy, sz):
        """Return box faces; used for batch rendering many boxes in one collection."""
        x0, x1 = cx - sx / 2.0, cx + sx / 2.0
        y0, y1 = cy - sy / 2.0, cy + sy / 2.0
        z0, z1 = cz - sz / 2.0, cz + sz / 2.0
        v = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        return [
            [v[i] for i in [0, 1, 2, 3]],
            [v[i] for i in [4, 5, 6, 7]],
            [v[i] for i in [0, 1, 5, 4]],
            [v[i] for i in [2, 3, 7, 6]],
            [v[i] for i in [1, 2, 6, 5]],
            [v[i] for i in [0, 3, 7, 4]],
        ]

    def _add_boxes_3d_batched(self, ax, boxes, edge_color=None, linewidth=0.0):
        """Draw many boxes as one Poly3DCollection.

        This is the main v15.1 performance fix. Matplotlib is slow when we add
        hundreds/thousands of individual Poly3DCollection objects. One large
        collection with many faces is far faster while preserving the same
        visible geometry.
        """
        if not boxes:
            return

        faces = []
        facecolors = []
        edgecolors = []

        for cx, cy, cz, sx, sy, sz, color, alpha in boxes:
            rgba = list(color)
            if len(rgba) < 4:
                rgba.append(alpha)
            else:
                rgba[3] = alpha

            box_faces = self._box_faces_3d_translated(cx, cy, cz, sx, sy, sz)
            faces.extend(box_faces)
            facecolors.extend([rgba] * len(box_faces))
            if edge_color is None:
                edgecolors.extend([rgba] * len(box_faces))
            else:
                edgecolors.extend([edge_color] * len(box_faces))

        poly = Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidth,
            alpha=None,
        )
        ax.add_collection3d(poly)

    def _draw_snapshot(self, idx:int):
        if self.result is None: self._draw_layout_preview(); return
        if idx<0 or idx>=len(self.result.snapshots) or self._drawing_snapshot: return
        self._drawing_snapshot=True
        try:
            snap=self.result.snapshots[idx]; temp=snap.temp_c; cfg=self.result.cfg; x=self.result.x_m*100; y=self.result.y_m*100
            self.fig.clear(); self.ax=self.fig.add_subplot(111)
            vmin,vmax=(self.vmin,self.vmax) if self.fixed_scale_var.get() and self.vmin is not None else (None,None)
            im=self.ax.imshow(temp.T,origin="lower",extent=[x[0],x[-1],y[0],y[-1]],aspect="equal",interpolation="bilinear",vmin=vmin,vmax=vmax)
            self.colorbar=self.fig.colorbar(im,ax=self.ax); self.colorbar.set_label("Temperature, °C")
            self._draw_fin_overlay_2d(self.ax, cfg)
            for r in cfg.resistors:
                l=r.length_mm/10; w=r.width_mm/10
                self.ax.add_patch(Rectangle((r.center_x_cm-l/2,r.center_y_cm-w/2),l,w,fill=False,linewidth=2)); self.ax.text(r.center_x_cm,r.center_y_cm,r.name,ha="center",va="center",fontsize=9)
            mi=np.unravel_index(np.argmax(temp),temp.shape); maxx=x[mi[0]]; maxy=y[mi[1]]; maxt=float(temp[mi]); avgt=float(np.mean(temp))
            self.ax.plot([maxx],[maxy],marker="x",markersize=10); self.ax.text(maxx,maxy,f" max {maxt:.1f}°C",va="bottom")
            self.ax.set_title(f"{snap.label} | avg {avgt:.1f}°C | max {maxt:.1f}°C"); self.ax.set_xlabel("x / width, cm"); self.ax.set_ylabel("y / height, cm"); self.ax.grid(True,alpha=.25)
            self.fig.subplots_adjust(left=.08,right=.88,bottom=.10,top=.92); self.canvas.draw_idle(); self.snapshot_label_var.set(f"{idx+1}/{len(self.result.snapshots)}: {snap.label}"); self._current_snapshot_idx=idx
        finally: self._drawing_snapshot=False
    def _slider_changed(self,value):
        if self._ignore_slider_callback or self._drawing_snapshot or self.result is None: return
        try: idx=int(round(float(value)))
        except Exception: return
        idx=max(0,min(len(self.result.snapshots)-1,idx))
        if self._current_snapshot_idx!=idx: self._draw_snapshot(idx)
    def _step_slider(self,d):
        if self.result is None: return
        cur=int(round(float(self.time_slider.get()))); new=max(0,min(len(self.result.snapshots)-1,cur+d))
        self._ignore_slider_callback=True; self.time_slider.set(new); self._ignore_slider_callback=False; self._draw_snapshot(new)
    def _redraw_current(self):
        if self.result is None: self._draw_layout_preview(); return
        self._current_snapshot_idx=None; self._draw_snapshot(int(round(float(self.time_slider.get()))))

    # -------------------------- simulation/optimizer --------------------------
    def _start_simulation(self):
        if self.worker_thread and self.worker_thread.is_alive(): messagebox.showinfo("Running","Already running."); return
        try: cfg=self._read_config()
        except Exception as e: messagebox.showerror("Invalid input",str(e)); return
        self.result=None; self.vmin=self.vmax=None; self.stop_event.clear(); self.run_button.configure(state="disabled"); self.cancel_button.configure(state="normal"); self.tabs.select(3); self._set_status("Starting simulation...")
        def worker():
            try: self.msg_queue.put(("done", run_simulation(cfg, progress_callback=lambda m:self.msg_queue.put(("progress",m)), stop_event=self.stop_event)))
            except Exception as e: self.msg_queue.put(("error",f"{e}\n\n{traceback.format_exc()}"))
        self.worker_thread=threading.Thread(target=worker,daemon=True); self.worker_thread.start()
    def _start_optimizer(self):
        if self.worker_thread and self.worker_thread.is_alive(): messagebox.showinfo("Running","Already running."); return
        try: cfg=self._read_config(); margin=parse_float(self.bank_margin_var.get(),"Margin",0.0); opt_grid=parse_float(self.optimizer_grid_var.get(),"Optimizer grid",3.0); depth=self.optimizer_depth_var.get()
        except Exception as e: messagebox.showerror("Invalid optimizer input",str(e)); return
        self.stop_event.clear(); self.cancel_button.configure(state="normal"); self.tabs.select(3); self._set_status(f"Deep optimizer running. Depth={depth}, optimizer grid={opt_grid:g} mm.")
        def worker():
            try:
                workers=int(parse_float(self.optimizer_workers_var.get(),"Workers",0))
                out=optimize_layout_deep(cfg=cfg, margin_cm=margin, optimizer_grid_mm=opt_grid, depth=depth, progress_callback=lambda m:self.msg_queue.put(("progress",m)), stop_event=self.stop_event, worker_count=workers)
                self.msg_queue.put(("opt_done",out))
            except Exception as e: self.msg_queue.put(("opt_error",f"{e}\n\n{traceback.format_exc()}"))
        self.worker_thread=threading.Thread(target=worker,daemon=True); self.worker_thread.start()
    def _cancel(self): self.stop_event.set(); self._append_status("Cancelling...")
    def _poll_queue(self):
        try:
            while True:
                typ,payload=self.msg_queue.get_nowait()
                if typ=="progress": self._set_progress(str(payload))
                elif typ=="done": self._simulation_done(payload)
                elif typ=="error": self.run_button.configure(state="normal"); self.cancel_button.configure(state="disabled"); self._append_status("Error:\n"+str(payload)); messagebox.showerror("Simulation error",str(payload).split("\n\n")[0])
                elif typ=="opt_done": self.cancel_button.configure(state="disabled"); self._apply_opt(*payload)
                elif typ=="opt_error": self.cancel_button.configure(state="disabled"); self._append_status("Optimizer error:\n"+str(payload)); messagebox.showerror("Optimizer error",str(payload).split("\n\n")[0])
        except queue.Empty: pass
        self.after(100,self._poll_queue)
    def _simulation_done(self,result):
        self.result=result; self.run_button.configure(state="normal"); self.cancel_button.configure(state="disabled")
        all_t=np.concatenate([s.temp_c.ravel() for s in result.snapshots]); self.vmin=float(np.min(all_t)); self.vmax=float(np.max(all_t))
        self.time_slider.configure(from_=0,to=max(0,len(result.snapshots)-1)); self._ignore_slider_callback=True; self.time_slider.set(0); self._ignore_slider_callback=False; self._current_snapshot_idx=None
        self._write_summary(result); self._draw_snapshot(0)
    def _apply_opt(self,positions,score,tried,details):
        for r,(x,y) in zip(self.resistors,positions): r.center_x_cm=float(x); r.center_y_cm=float(y)
        self._refresh_resistor_tree(); lines=[f"Optimizer finished. Tried/evaluated about {tried} layouts/moves.",f"Score: {score:.3f} lower is better."]
        if details and details.get("final_plate_max_c") is not None: lines.append(f"Fine-grid estimated max: {details.get('final_plate_max_c'):.1f} °C")
        lines.append("Positions applied. Run full simulation for final result."); self._set_status("\n".join(lines)); self._draw_layout_preview()

    # ------------------------------- summary/tools -------------------------------
    def _heatsink_summary_line(self, s: Dict) -> str:
        hs=s.get('heatsink_geometry') or {}
        if hs.get('enabled'):
            return (
                f"Geometry fins: {hs.get('fin_count',0)} fins, raw {hs.get('raw_fin_area_cm2',0):.0f} cm², "
                f"avg η {hs.get('average_fin_efficiency_percent',0):.0f}%"
            )
        return "Geometry fins: off"
    def _write_summary(self,result):
        s = result.summary
        snap = result.snapshots[-1]
        temp = snap.temp_c
        max_plate = float(np.max(temp))
        avg_plate = float(np.mean(temp))

        max_lim = self._f(self.max_plate_temp_var.get(), 90)
        case_lim = self._f(self.max_res_case_temp_var.get(), 120)
        element_lim = self._f(self.max_res_element_temp_var.get(), 200)
        margin = self._f(self.cooling_margin_var.get(), 25)

        safe_h = result.cfg.convection_h_w_m2k * (1 - max(0, min(90, margin)) / 100)
        conservative = result.cfg.ambient_c + s['total_power_w'] / (
            max(.1, safe_h) * (s['plate_area_both_faces_cm2'] / 10000)
        )

        hot = s.get("hottest_plate_point", {})
        lines = [
            "Simulation complete.",
            "",
            f"Grid: {s['grid_cells_x']} × {s['grid_cells_y']}",
            f"Total heat: {s['total_power_w']:.2f} W",
            f"Plate area both faces: {s['plate_area_both_faces_cm2']:.0f} cm²",
            f"Base h: {s.get('base_h_w_m2k', result.cfg.convection_h_w_m2k):.2f} W/m²K",
            f"Effective h used: {s.get('effective_convection_h_w_m2k', result.cfg.convection_h_w_m2k):.2f} W/m²K",
            f"Heatsink effective extra area: {s.get('heatsink_effective_extra_area_cm2', 0):.0f} cm²",
            self._heatsink_summary_line(s),
            f"Final endpoint: {snap.label}",
            f"Average plate temp: {avg_plate:.1f} °C",
            f"Max plate temp: {max_plate:.1f} °C",
        ]

        if hot:
            lines.append(f"Hottest plate point: x={hot.get('x_cm',0):.1f} cm, y={hot.get('y_cm',0):.1f} cm")

        lines += [
            "",
            "Pass/fail:",
            f"  Plate: {'PASS' if max_plate <= max_lim else 'WARNING' if max_plate <= max_lim + 15 else 'FAIL'} ({max_plate:.1f} °C vs limit {max_lim:.1f} °C)",
            f"  Conservative average with {margin:.0f}% h margin: {conservative:.1f} °C",
            "",
            "Resistor temperature stack:",
            f"  Case→plate: {s.get('resistor_case_to_plate_cw', result.cfg.resistor_case_to_plate_cw):g} °C/W",
            f"  Element→case: {s.get('resistor_element_to_case_cw', result.cfg.resistor_element_to_case_cw):g} °C/W",
        ]

        for rr in s.get("resistors", []):
            case_max = rr.get("estimated_case_max_temp_c", rr.get("final_max_temp_c", 0))
            elem_max = rr.get("estimated_element_max_temp_c", case_max)
            case_status = 'PASS' if case_max <= case_lim else 'WARNING' if case_max <= case_lim + 15 else 'FAIL'
            elem_status = 'PASS' if elem_max <= element_lim else 'WARNING' if elem_max <= element_lim + 25 else 'FAIL'
            lines.append(f"  {rr.get('name','R?')}:")
            lines.append(
                f"      plate footprint avg {rr.get('plate_footprint_avg_temp_c',0):.1f} °C, "
                f"max {rr.get('plate_footprint_max_temp_c',0):.1f} °C"
            )
            lines.append(f"      estimated case/body max {case_max:.1f} °C: {case_status}")
            lines.append(f"      estimated internal element max {elem_max:.1f} °C: {elem_status}")

        if s.get("precision_notes"):
            lines += ["", "Precision notes:"]
            for note in s["precision_notes"]:
                lines.append(f"  - {note}")

        lines += [
            "",
            "Warm-up rough average:",
            f"  Time constant: {format_time(s['tau_s'])}",
            f"  ~90% final average: {format_time(s['t90_s'])}",
            f"  ~95% final average: {format_time(s['t95_s'])}",
            "",
            "Model limit: airflow, mounting pressure, and actual resistor datasheet values should still be verified with measurement.",
        ]

        self._set_status("\n".join(lines))
    def _f(self,t,fb):
        try: return float(str(t).replace(",","."))
        except Exception: return fb
    def _calc_electrical(self):
        try:
            v=parse_float(self.e_supply_v_var.get(),"Supply",0); r=parse_float(self.e_res_ohm_var.get(),"Resistance",1e-9); n=int(parse_float(self.e_count_var.get(),"Count",1)); conn=self.e_connection_var.get()
            if conn=="Series": total_r=n*r; current=v/total_r; each=current*current*r
            else: total_r=r/n; current=v/total_r; each=v*v/r
            total=each*n; self.e_last_power_each=each; warn="\nWARNING: above 50 W per resistor." if each>50 else ""
            self.e_result_var.set(f"Total R: {total_r:.4g} Ω\nTotal current: {current:.2f} A\nPower each: {each:.2f} W\nTotal heat: {total:.2f} W"+warn)
        except Exception as e: self.e_result_var.set(f"Error: {e}")
    def _apply_electrical(self):
        if self.e_last_power_each is None: self._calc_electrical()
        if self.e_last_power_each is None: return
        self.bank_power_var.set(f"{self.e_last_power_each:.3g}")
        for r in self.resistors: r.power_w=float(self.e_last_power_each)
        self._refresh_resistor_tree(); self._set_status(f"Applied {self.e_last_power_each:.2f} W to current resistors/bank power.")
    def _calc_size(self):
        try:
            p=parse_float(self.size_power_var.get(),"Power",0); target=parse_float(self.size_target_var.get(),"Target"); amb=parse_float(self.size_ambient_var.get(),"Ambient"); h=parse_float(self.size_h_var.get(),"h",.1); rise=target-amb
            if rise<=0: raise ValueError("Target must be above ambient.")
            area_cm2=p/(h*rise)*10000; square=math.sqrt(area_cm2/2)
            self.size_result_var.set(f"Required area both faces: {area_cm2:.0f} cm²\nApprox square plate: {square:.1f} × {square:.1f} cm\nUse larger for hotspots/enclosures.")
        except Exception as e: self.size_result_var.set(f"Error: {e}")
    def _calc_h_cal(self):
        try:
            p=parse_float(self.cal_power_var.get(),"Power",0); area=parse_float(self.cal_area_var.get(),"Area",.1); amb=parse_float(self.cal_ambient_var.get(),"Ambient"); meas=parse_float(self.cal_measured_var.get(),"Measured"); rise=meas-amb
            if rise<=0: raise ValueError("Measured must be above ambient.")
            self._calibrated_h_value=p/((area/10000)*rise); self.cal_result_var.set(f"Estimated h ≈ {self._calibrated_h_value:.2f} W/m²K")
        except Exception as e: self._calibrated_h_value=None; self.cal_result_var.set(f"Error: {e}")
    def _use_cal_h(self):
        if self._calibrated_h_value is None: self._calc_h_cal()
        if self._calibrated_h_value is not None: self.advanced_cooling_var.set(False); self.h_var.set(f"{self._calibrated_h_value:.2f}"); self._update_h_label()

    # ------------------------------- save/load -------------------------------
    def _save_config(self):
        try: cfg=self._read_config()
        except Exception as e: messagebox.showerror("Invalid input",str(e)); return
        path=filedialog.asksaveasfilename(title="Save config",defaultextension=".json",filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        data=asdict(cfg); data['ui_safety']={
            'max_plate': self.max_plate_temp_var.get(),
            'max_case': self.max_res_case_temp_var.get(),
            'max_element': self.max_res_element_temp_var.get(),
            'extra_cw': self.extra_cw_var.get(),
            'element_cw': self.element_cw_var.get(),
            'margin': self.cooling_margin_var.get()
        }
        Path(path).write_text(json.dumps(data,indent=2),encoding="utf-8"); self._append_status(f"Saved config: {path}")
    def _load_config(self):
        path=filedialog.askopenfilename(title="Load config",filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        try:
            data=json.loads(Path(path).read_text(encoding="utf-8")); safety=data.pop('ui_safety',{})
            data['resistors']=[Resistor(**r) for r in data['resistors']]; cfg=PlateConfig(**data); self._apply_config(cfg)
            self.max_plate_temp_var.set(str(safety.get('max_plate', self.max_plate_temp_var.get())))
            self.max_res_case_temp_var.set(str(safety.get('max_case', self.max_res_case_temp_var.get())))
            self.max_res_element_temp_var.set(str(safety.get('max_element', self.max_res_element_temp_var.get())))
            self.extra_cw_var.set(str(safety.get('extra_cw', getattr(cfg, 'resistor_case_to_plate_cw', self.extra_cw_var.get()))))
            self.element_cw_var.set(str(safety.get('element_cw', getattr(cfg, 'resistor_element_to_case_cw', self.element_cw_var.get()))))
            self.cooling_margin_var.set(str(safety.get('margin', self.cooling_margin_var.get())))
            self._append_status(f"Loaded config: {path}")
        except Exception as e: messagebox.showerror("Load failed",str(e))
    def _apply_config(self,cfg):
        self.plate_width_x_var.set(f"{cfg.plate_length_cm:g}"); self.plate_height_y_var.set(f"{cfg.plate_width_cm:g}"); self.plate_thickness_var.set(f"{cfg.plate_thickness_mm:g}"); self.material_var.set(cfg.material_name); self.k_var.set(f"{cfg.thermal_conductivity_w_mk:g}"); self.rho_var.set(f"{cfg.density_kg_m3:g}"); self.cp_var.set(f"{cfg.heat_capacity_j_kgk:g}"); self.ambient_var.set(f"{cfg.ambient_c:g}"); self.h_var.set(f"{cfg.convection_h_w_m2k:g}"); self.grid_var.set(f"{cfg.grid_mm:g}"); self.initial_temp_var.set(f"{cfg.initial_plate_temp_c:g}"); self.max_time_var.set(format_time(cfg.max_time_s).replace(' ','')); self.snapshot_every_var.set(format_time(cfg.snapshot_every_s).replace(' ','')); self.include_steady_var.set(cfg.include_steady_state); self.advanced_cooling_var.set(getattr(cfg,'advanced_cooling_enabled',False)); self.orientation_var.set(display_from_key(ORIENTATION_OPTIONS,getattr(cfg,'orientation','vertical'))); self.environment_var.set(display_from_key(ENVIRONMENT_OPTIONS,getattr(cfg,'environment','open_air'))); self.clearance_var.set(f"{getattr(cfg,'wall_clearance_cm',20):g}"); self.surface_var.set(display_from_key(SURFACE_OPTIONS,getattr(cfg,'surface_finish','bare_metal'))); self.air_movement_var.set(display_from_key(AIR_MOVEMENT_OPTIONS,getattr(cfg,'air_movement','still_air'))); self.hot_air_path_var.set(display_from_key(HOT_AIR_PATH_OPTIONS,getattr(cfg,'hot_air_path','free_rise'))); self.blockage_var.set(f"{getattr(cfg,'blockage_percent',0):g}"); self.heatsink_enabled_var.set(getattr(cfg,'heatsink_enabled',False)); self.heatsink_mode_var.set("Geometry builder" if getattr(cfg,'heatsink_geometry_enabled',False) else "Simple extra area"); self.heatsink_area_var.set(f"{getattr(cfg,'heatsink_extra_area_cm2',0):g}"); self.heatsink_eff_var.set(f"{getattr(cfg,'heatsink_efficiency_percent',70):g}"); self.heatsink_hmul_var.set(f"{getattr(cfg,'heatsink_h_multiplier',1):g}"); self.heatsink_fin_orientation_var.set("Fins run along X, spread across Y" if getattr(cfg,'heatsink_fin_orientation','run_y')=='run_x' else "Fins run along Y, spread across X"); self.heatsink_fin_count_var.set(f"{getattr(cfg,'heatsink_fin_count',0):g}"); self.heatsink_fin_thickness_var.set("same" if getattr(cfg,'heatsink_fin_thickness_mm',0)<=0 else f"{getattr(cfg,'heatsink_fin_thickness_mm',0):g}"); self.heatsink_fin_height_var.set(f"{getattr(cfg,'heatsink_fin_default_height_mm',30):g}"); self.heatsink_fin_run_length_var.set("full" if getattr(cfg,'heatsink_fin_run_length_cm',0)<=0 else f"{getattr(cfg,'heatsink_fin_run_length_cm',0):g}"); self.heatsink_fin_positions_var.set(getattr(cfg,'heatsink_fin_positions_cm','even')); self.heatsink_fin_heights_var.set(getattr(cfg,'heatsink_fin_heights_mm','same')); self.heatsink_fin_segments_var.set("auto" if getattr(cfg,'heatsink_fin_segments',0)<=0 else f"{getattr(cfg,'heatsink_fin_segments',0):g}"); self.extra_cw_var.set(f"{getattr(cfg,'resistor_case_to_plate_cw', self._f(self.extra_cw_var.get(),0.5)):g}"); self.element_cw_var.set(f"{getattr(cfg,'resistor_element_to_case_cw', self._f(self.element_cw_var.get(),0.6)):g}"); self.resistors=list(cfg.resistors); self._refresh_resistor_tree(); self._update_material_fields(); self._update_h_label(); self._update_heatsink_label(); self._draw_layout_preview()
    def _current_snapshot(self):
        if self.result is None: return None
        idx=int(round(float(self.time_slider.get()))); return self.result.snapshots[idx] if 0<=idx<len(self.result.snapshots) else None
    def _export_image(self):
        if self.result is None: messagebox.showinfo("No result","Run a simulation first."); return
        snap=self._current_snapshot(); default='heatmap.png' if snap is None else f"heatmap_{safe_time_name(snap.label)}.png"
        path=filedialog.asksaveasfilename(title="Export heatmap",initialfile=default,defaultextension='.png',filetypes=[('PNG','*.png'),('All','*.*')])
        if path: self.fig.savefig(path,dpi=180); self._append_status(f"Exported image: {path}")
    def _export_csv(self):
        if self.result is None: messagebox.showinfo("No result","Run a simulation first."); return
        snap=self._current_snapshot();
        if snap is None: return
        path=filedialog.asksaveasfilename(title="Export grid",initialfile=f"temperature_grid_{safe_time_name(snap.label)}.csv",defaultextension='.csv',filetypes=[('CSV','*.csv'),('All','*.*')])
        if path: save_temperature_grid_csv(self.result.x_m,self.result.y_m,snap.temp_c,Path(path)); self._append_status(f"Exported CSV: {path}")

    # ----------------------------- 3D viewer / fin designer -----------------------------
    def _current_3d_snapshot_data(self):
        """Return cfg, x_cm, y_cm, temp array, label for the 3D viewer."""
        if self.result is not None:
            snap = self._current_snapshot() or self.result.snapshots[0]
            return self.result.cfg, self.result.x_m * 100.0, self.result.y_m * 100.0, snap.temp_c, snap.label

        cfg = self._read_config_loose_for_heatsink()
        nx = max(8, min(60, int(round(cfg.plate_length_cm / max(0.5, cfg.grid_mm / 10.0)))))
        ny = max(8, min(60, int(round(cfg.plate_width_cm / max(0.5, cfg.grid_mm / 10.0)))))
        x = np.linspace(-cfg.plate_length_cm / 2.0, cfg.plate_length_cm / 2.0, nx)
        y = np.linspace(-cfg.plate_width_cm / 2.0, cfg.plate_width_cm / 2.0, ny)
        temp = np.full((nx, ny), cfg.ambient_c, dtype=float)
        return cfg, x, y, temp, "live geometry"

    def _temp_nearest_3d(self, x_cm, y_cm, xs_cm, ys_cm, temp):
        ix = int(np.argmin(np.abs(xs_cm - x_cm)))
        iy = int(np.argmin(np.abs(ys_cm - y_cm)))
        return float(temp[ix, iy])

    def _heatmap_cmap(self):
        """Return the same colormap used by the 2D heatmap view."""
        # The 2D imshow view uses Matplotlib's default colormap unless specified.
        # In standard Matplotlib that is 'viridis', and cm.get_cmap() returns
        # the active default colormap object.
        return cm.get_cmap()

    def _heatmap_norm_limits_for_temp(self, temp):
        """Use the same temperature scaling rule as the 2D view."""
        if self.fixed_scale_var.get() and self.vmin is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
        else:
            vmin = float(np.min(temp))
            vmax = float(np.max(temp))
            if vmax <= vmin:
                vmax = vmin + 1.0
        return vmin, vmax

    def _box_faces_3d(self, cx, cy, cz, sx, sy, sz):
        x0, x1 = cx - sx / 2.0, cx + sx / 2.0
        y0, y1 = cy - sy / 2.0, cy + sy / 2.0
        z0, z1 = cz - sz / 2.0, cz + sz / 2.0
        v = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        return [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],[v[1],v[2],v[6],v[5]],[v[2],v[3],v[7],v[6]],[v[3],v[0],v[4],v[7]]]

    def _add_box_3d(self, ax, cx, cy, cz, sx, sy, sz, color, alpha=0.55, edge=True):
        poly = Poly3DCollection(self._box_faces_3d(cx, cy, cz, sx, sy, sz), facecolors=color, edgecolors=("k" if edge else color), linewidths=0.35, alpha=alpha)
        ax.add_collection3d(poly)

    def _add_box_edges_3d(self, ax, cx, cy, cz, sx, sy, sz, color="k", linewidth=0.9):
        """Add wire edges for a 3D box.

        Matplotlib's mplot3d can depth-sort filled polygons poorly. Drawing
        resistor outlines after solids makes them readable from more angles.
        """
        faces = self._box_faces_3d(cx, cy, cz, sx, sy, sz)
        edges = set()
        for face in faces:
            for i in range(len(face)):
                a = face[i]
                b = face[(i + 1) % len(face)]
                edges.add(tuple(sorted((a, b))))
        for a, b in edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=linewidth)

    def _open_3d_viewer(self):
        """Open the v15.2 canvas 3D viewer.

        This intentionally does not use Matplotlib mplot3d. The old viewer was
        bottlenecked by 3D artist creation/depth sorting. This viewer projects
        the model onto a Tk Canvas and paints polygons directly, which is much
        closer to what this simulator actually needs: fast engineering
        visualization of a known geometry.
        """
        win = tk.Toplevel(self)
        win.title("Fast 3D engineering viewer")
        win.geometry("1180x820")
        win.transient(self)

        controls = ttk.Frame(win, padding=8)
        controls.pack(side="top", fill="x")

        show_fins = tk.BooleanVar(value=True)
        show_res = tk.BooleanVar(value=True)
        show_plate_heat = tk.BooleanVar(value=True)
        transparent_plate = tk.BooleanVar(value=False)
        exploded_view = tk.BooleanVar(value=True)
        show_edges = tk.BooleanVar(value=True)

        side_var = tk.StringVar(value=self.resistor_side_var.get())
        view_var = tk.StringVar(value="Isometric")
        snap_var = tk.IntVar(value=int(round(float(self.time_slider.get()))) if self.result is not None else 0)
        yaw_var = tk.DoubleVar(value=-45.0)
        elev_var = tk.DoubleVar(value=35.0)
        zoom_var = tk.DoubleVar(value=1.0)

        ttk.Checkbutton(controls, text="Plate heat", variable=show_plate_heat).pack(side="left", padx=(0,8))
        ttk.Checkbutton(controls, text="Show fins", variable=show_fins).pack(side="left", padx=(0,8))
        ttk.Checkbutton(controls, text="Show resistors", variable=show_res).pack(side="left", padx=(0,8))
        ttk.Checkbutton(controls, text="Edges", variable=show_edges).pack(side="left", padx=(0,8))
        ttk.Checkbutton(controls, text="Light plate", variable=transparent_plate).pack(side="left", padx=(0,8))
        ttk.Checkbutton(controls, text="Exploded", variable=exploded_view).pack(side="left", padx=(0,8))

        ttk.Label(controls, text="Resistors").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=side_var,
            values=["Front / flat side", "Back / fin side", "Both sides"],
            state="readonly",
            width=16,
        ).pack(side="left", padx=(4,8))

        ttk.Label(controls, text="View").pack(side="left")
        view_combo = ttk.Combobox(
            controls,
            textvariable=view_var,
            values=["Isometric", "Top", "Front", "Back", "Side X", "Side Y"],
            state="readonly",
            width=10,
        )
        view_combo.pack(side="left", padx=(4,8))

        label_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=label_var, width=42).pack(side="right")

        sliders = ttk.Frame(win, padding=(8, 0, 8, 6))
        sliders.pack(side="top", fill="x")

        max_snap = len(self.result.snapshots)-1 if self.result is not None else 0
        ttk.Label(sliders, text="Time").grid(row=0, column=0, sticky="w")
        time_slider = ttk.Scale(sliders, from_=0, to=max_snap, orient="horizontal", variable=snap_var)
        time_slider.grid(row=0, column=1, sticky="ew", padx=(6,12))

        ttk.Label(sliders, text="Yaw").grid(row=0, column=2, sticky="w")
        yaw_slider = ttk.Scale(sliders, from_=-180, to=180, orient="horizontal", variable=yaw_var)
        yaw_slider.grid(row=0, column=3, sticky="ew", padx=(6,12))

        ttk.Label(sliders, text="Elev").grid(row=0, column=4, sticky="w")
        elev_slider = ttk.Scale(sliders, from_=5, to=89, orient="horizontal", variable=elev_var)
        elev_slider.grid(row=0, column=5, sticky="ew", padx=(6,12))

        ttk.Label(sliders, text="Zoom").grid(row=0, column=6, sticky="w")
        zoom_slider = ttk.Scale(sliders, from_=0.35, to=2.5, orient="horizontal", variable=zoom_var)
        zoom_slider.grid(row=0, column=7, sticky="ew", padx=(6,0))

        for col in (1, 3, 5, 7):
            sliders.columnconfigure(col, weight=1)

        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(side="top", fill="both", expand=True)

        cnv = tk.Canvas(canvas_frame, background="white", highlightthickness=0)
        cnv.pack(side="left", fill="both", expand=True)

        info = tk.Text(canvas_frame, width=30, wrap="word")
        info.pack(side="right", fill="y")
        info.insert(
            "end",
            """v15.2 renderer

This viewer uses a custom Tk Canvas projection instead of Matplotlib mplot3d.

It keeps fin segment/detail geometry, but draws it as direct 2D polygons.
That is much faster for this kind of engineering view.

Use mouse drag to rotate.
Mouse wheel zooms.
Time slider follows simulation snapshots.
""",
        )
        info.configure(state="disabled")

        state = {"drag_x": None, "drag_y": None, "last_draw": 0.0}

        def rgba_to_hex(rgba, lighten=0.0):
            r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
            if lighten:
                r = r * (1.0 - lighten) + lighten
                g = g * (1.0 - lighten) + lighten
                b = b * (1.0 - lighten) + lighten
            r = max(0, min(255, int(round(r * 255))))
            g = max(0, min(255, int(round(g * 255))))
            b = max(0, min(255, int(round(b * 255))))
            return f"#{r:02x}{g:02x}{b:02x}"

        def temp_color(temp_value, cmap, norm, lighten=0.0):
            return rgba_to_hex(cmap(norm(float(temp_value))), lighten=lighten)

        def edges_from_centers(vals):
            vals = np.asarray(vals, dtype=float)
            if len(vals) == 1:
                return np.array([vals[0] - 0.5, vals[0] + 0.5])
            mids = (vals[:-1] + vals[1:]) / 2.0
            first = vals[0] - (mids[0] - vals[0])
            last = vals[-1] + (vals[-1] - mids[-1])
            return np.concatenate([[first], mids, [last]])

        def project_point(p, yaw_deg, elev_deg, scale, cx, cy):
            x, y, z = p
            yaw = math.radians(yaw_deg)
            elev = math.radians(elev_deg)

            # Rotate in plate plane.
            x1 = x * math.cos(yaw) - y * math.sin(yaw)
            y1 = x * math.sin(yaw) + y * math.cos(yaw)

            # Orthographic camera elevation. elev=90 is top-down.
            screen_x = cx + x1 * scale
            screen_y = cy - (y1 * math.sin(elev) + z * math.cos(elev)) * scale

            # Larger depth is closer to viewer.
            depth = y1 * math.cos(elev) - z * math.sin(elev)
            return screen_x, screen_y, depth

        def box_faces(cx, cy, cz, sx, sy, sz):
            x0, x1 = cx - sx / 2.0, cx + sx / 2.0
            y0, y1 = cy - sy / 2.0, cy + sy / 2.0
            z0, z1 = cz - sz / 2.0, cz + sz / 2.0
            v = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ]
            return [
                [v[i] for i in (0,1,2,3)],
                [v[i] for i in (4,5,6,7)],
                [v[i] for i in (0,1,5,4)],
                [v[i] for i in (1,2,6,5)],
                [v[i] for i in (2,3,7,6)],
                [v[i] for i in (3,0,4,7)],
            ]

        def add_box(objects, cx, cy, cz, sx, sy, sz, fill, outline=""):
            for face in box_faces(cx, cy, cz, sx, sy, sz):
                objects.append({
                    "kind": "poly",
                    "points": face,
                    "fill": fill,
                    "outline": outline,
                    "width": 1 if outline else 0,
                })

        def add_plate_heatmap(objects, cfg, x_cm, y_cm, temp, cmap, norm):
            x_edges = edges_from_centers(x_cm)
            y_edges = edges_from_centers(y_cm)

            # Drawing every solver cell is usually fine, but very high grids can
            # create tens of thousands of canvas polygons. Cap display cells; the
            # 2D heatmap remains the precise view for the temperature field.
            max_cells_axis = 75
            sx = max(1, int(math.ceil(len(x_cm) / max_cells_axis)))
            sy = max(1, int(math.ceil(len(y_cm) / max_cells_axis)))

            lighten = 0.32 if transparent_plate.get() else 0.0

            for ix in range(0, len(x_cm), sx):
                ix2 = min(len(x_cm), ix + sx)
                for iy in range(0, len(y_cm), sy):
                    iy2 = min(len(y_cm), iy + sy)
                    tval = float(np.mean(temp[ix:ix2, iy:iy2]))
                    fill = temp_color(tval, cmap, norm, lighten=lighten)
                    face = [
                        (float(x_edges[ix]),  float(y_edges[iy]),  0.0),
                        (float(x_edges[ix2]), float(y_edges[iy]),  0.0),
                        (float(x_edges[ix2]), float(y_edges[iy2]), 0.0),
                        (float(x_edges[ix]),  float(y_edges[iy2]), 0.0),
                    ]
                    objects.append({"kind": "poly", "points": face, "fill": fill, "outline": "", "width": 0})

        def add_fins(objects, cfg, x_cm, y_cm, temp, cmap, norm):
            t_cm = max(0.02, cfg.plate_thickness_mm / 10.0)
            fin_segments = heatsink_fin_segment_specs(cfg)
            height_slices = 8

            for fin in fin_segments:
                fx = fin["center_x_cm"]
                fy = fin["center_y_cm"]
                fw = max(0.02, fin["footprint_x_cm"])
                fr = max(0.02, fin["footprint_y_cm"])
                fh = max(0.02, fin["height_mm"] / 10.0)

                base_tc = self._temp_nearest_3d(fx, fy, x_cm, y_cm, temp)
                dz = fh / height_slices

                for zi in range(height_slices):
                    frac = (zi + 0.5) / height_slices
                    fin_tc = fin_temperature_at_height(cfg, fin, base_tc, frac)
                    zc = -t_cm - dz * (zi + 0.5)
                    fill = temp_color(fin_tc, cmap, norm)
                    add_box(objects, fx, fy, zc, fw, fr, dz, fill, "")

        def add_resistors(objects, labels, cfg, x_cm, y_cm, temp, cmap, norm):
            t_cm = max(0.02, cfg.plate_thickness_mm / 10.0)
            side = side_var.get()
            explode_gap = 0.72 if exploded_view.get() else 0.0
            z_sides = []

            if side in ("Front / flat side", "Both sides"):
                z_sides.append((0.28 + explode_gap, 1))
            if side in ("Back / fin side", "Both sides"):
                z_sides.append((-t_cm - 0.28 - explode_gap, -1))

            for r in cfg.resistors:
                sx = max(0.03, r.length_mm / 10.0)
                sy = max(0.03, r.width_mm / 10.0)
                sz = 0.55
                tc = self._temp_nearest_3d(r.center_x_cm, r.center_y_cm, x_cm, y_cm, temp)
                case_tc = tc + r.power_w * max(0.0, float(getattr(cfg, "resistor_case_to_plate_cw", 0.5)))
                fill = temp_color(case_tc, cmap, norm)
                outline = "black" if show_edges.get() else ""

                for zc, sign in z_sides:
                    add_box(objects, r.center_x_cm, r.center_y_cm, zc, sx, sy, sz, fill, outline)
                    labels.append((r.center_x_cm, r.center_y_cm, zc + sign * 0.45, r.name))

        def draw_colorbar(cnv, cmap, norm, width, height):
            bar_w = 18
            x0 = width - 70
            y0 = 70
            bar_h = max(120, min(300, height - 180))
            steps = 80
            for i in range(steps):
                frac0 = i / steps
                frac1 = (i + 1) / steps
                value = norm.vmax - frac0 * (norm.vmax - norm.vmin)
                fill = temp_color(value, cmap, norm)
                ya = y0 + frac0 * bar_h
                yb = y0 + frac1 * bar_h
                cnv.create_rectangle(x0, ya, x0 + bar_w, yb, outline="", fill=fill)
            cnv.create_rectangle(x0, y0, x0 + bar_w, y0 + bar_h, outline="black")
            cnv.create_text(x0 + bar_w + 5, y0, anchor="w", text=f"{norm.vmax:.1f}°C", font=("TkDefaultFont", 8))
            cnv.create_text(x0 + bar_w + 5, y0 + bar_h, anchor="w", text=f"{norm.vmin:.1f}°C", font=("TkDefaultFont", 8))

        def apply_view_preset(*_):
            view = view_var.get()
            if view == "Top":
                yaw_var.set(0.0); elev_var.set(89.0)
            elif view == "Front":
                yaw_var.set(0.0); elev_var.set(20.0)
            elif view == "Back":
                yaw_var.set(180.0); elev_var.set(20.0)
            elif view == "Side X":
                yaw_var.set(90.0); elev_var.set(20.0)
            elif view == "Side Y":
                yaw_var.set(0.0); elev_var.set(20.0)
            else:
                yaw_var.set(-45.0); elev_var.set(35.0)
            draw()

        def draw(*_):
            try:
                cfg, x_cm, y_cm, temp, label = self._current_3d_snapshot_data()
                if self.result is not None:
                    idx = max(0, min(len(self.result.snapshots)-1, int(round(float(snap_var.get())))))
                    snap = self.result.snapshots[idx]
                    cfg = self.result.cfg
                    x_cm = self.result.x_m * 100.0
                    y_cm = self.result.y_m * 100.0
                    temp = snap.temp_c
                    label = snap.label

                width = max(100, cnv.winfo_width())
                height = max(100, cnv.winfo_height())
                cnv.delete("all")

                cmap = self._heatmap_cmap()
                vmin, vmax = self._heatmap_norm_limits_for_temp(temp)
                norm = Normalize(vmin=vmin, vmax=vmax)

                maxx = max(abs(float(x_cm[0])), abs(float(x_cm[-1])), 1.0)
                maxy = max(abs(float(y_cm[0])), abs(float(y_cm[-1])), 1.0)
                max_fin_h = max([fin.get("height_mm", 0) / 10.0 for fin in heatsink_fin_specs(cfg)] + [1.0])
                geom_span = max(2 * maxx, 2 * maxy, max_fin_h + 4.0)
                scale = 0.78 * min(width - 120, height - 40) / max(1.0, geom_span) * float(zoom_var.get())
                cx = width * 0.48
                cy = height * 0.52

                objects = []
                labels = []

                if show_plate_heat.get():
                    add_plate_heatmap(objects, cfg, x_cm, y_cm, temp, cmap, norm)

                if show_fins.get():
                    add_fins(objects, cfg, x_cm, y_cm, temp, cmap, norm)

                if show_res.get():
                    add_resistors(objects, labels, cfg, x_cm, y_cm, temp, cmap, norm)

                yaw = float(yaw_var.get())
                elev = float(elev_var.get())

                rendered = []
                for obj in objects:
                    pts = [project_point(p, yaw, elev, scale, cx, cy) for p in obj["points"]]
                    coords = []
                    for sx, sy, _depth in pts:
                        coords.extend([sx, sy])
                    depth = sum(p[2] for p in pts) / max(1, len(pts))
                    rendered.append((depth, coords, obj))

                # Painter sort: far to near.
                rendered.sort(key=lambda item: item[0])

                for _depth, coords, obj in rendered:
                    cnv.create_polygon(
                        coords,
                        fill=obj["fill"],
                        outline=obj["outline"],
                        width=obj.get("width", 0),
                    )

                # Draw labels last.
                for lx, ly, lz, txt in labels:
                    sx, sy, _d = project_point((lx, ly, lz), yaw, elev, scale, cx, cy)
                    cnv.create_text(sx, sy, text=txt, fill="black", font=("TkDefaultFont", 8, "bold"))

                # Simple axes / orientation cross.
                origin = project_point((0,0,0), yaw, elev, scale, cx, cy)
                axx = project_point((min(5, maxx),0,0), yaw, elev, scale, cx, cy)
                axy = project_point((0,min(5, maxy),0), yaw, elev, scale, cx, cy)
                axz = project_point((0,0,min(3, max_fin_h)), yaw, elev, scale, cx, cy)
                cnv.create_line(origin[0], origin[1], axx[0], axx[1], fill="#333", width=2)
                cnv.create_line(origin[0], origin[1], axy[0], axy[1], fill="#333", width=2)
                cnv.create_line(origin[0], origin[1], axz[0], axz[1], fill="#333", width=2)
                cnv.create_text(axx[0], axx[1], text="x", fill="#333")
                cnv.create_text(axy[0], axy[1], text="y", fill="#333")
                cnv.create_text(axz[0], axz[1], text="z", fill="#333")

                draw_colorbar(cnv, cmap, norm, width, height)

                fin_seg_count = len(heatsink_fin_segment_specs(cfg))
                label_var.set(f"{label} | Canvas renderer | fin segs: {fin_seg_count} | objects: {len(rendered)}")
            except Exception as e:
                label_var.set(f"3D error: {e}")

        def throttled_draw(*_):
            # Tk Canvas redraws are cheap, but sliders can fire very rapidly.
            # This coalesces bursts into one redraw per event loop cycle.
            if state.get("pending"):
                return
            state["pending"] = True
            def run():
                state["pending"] = False
                draw()
            win.after_idle(run)

        def on_press(event):
            state["drag_x"] = event.x
            state["drag_y"] = event.y

        def on_drag(event):
            if state["drag_x"] is None:
                return
            dx = event.x - state["drag_x"]
            dy = event.y - state["drag_y"]
            state["drag_x"] = event.x
            state["drag_y"] = event.y
            yaw_var.set(float(yaw_var.get()) + dx * 0.45)
            elev_var.set(max(5.0, min(89.0, float(elev_var.get()) - dy * 0.35)))
            throttled_draw()

        def on_release(event):
            state["drag_x"] = None
            state["drag_y"] = None

        def on_wheel(event):
            delta = 1 if getattr(event, "delta", 0) > 0 else -1
            if getattr(event, "num", None) == 4:
                delta = 1
            elif getattr(event, "num", None) == 5:
                delta = -1
            zoom = float(zoom_var.get()) * (1.10 if delta > 0 else 0.90)
            zoom_var.set(max(0.35, min(2.5, zoom)))
            throttled_draw()

        for var in (show_fins, show_res, show_plate_heat, transparent_plate, exploded_view, show_edges, side_var, yaw_var, elev_var, zoom_var, snap_var):
            var.trace_add("write", lambda *a: throttled_draw())

        view_var.trace_add("write", apply_view_preset)
        time_slider.configure(command=lambda v: throttled_draw())
        yaw_slider.configure(command=lambda v: throttled_draw())
        elev_slider.configure(command=lambda v: throttled_draw())
        zoom_slider.configure(command=lambda v: throttled_draw())

        cnv.bind("<ButtonPress-1>", on_press)
        cnv.bind("<B1-Motion>", on_drag)
        cnv.bind("<ButtonRelease-1>", on_release)
        cnv.bind("<MouseWheel>", on_wheel)
        cnv.bind("<Button-4>", on_wheel)
        cnv.bind("<Button-5>", on_wheel)
        cnv.bind("<Configure>", lambda e: throttled_draw())

        apply_view_preset()


    def _fin_count_int(self):
        return max(0, int(round(parse_float(self.heatsink_fin_count_var.get(), "Fin count", 0))))

    def _fin_across_bounds(self):
        px=parse_float(self.plate_width_x_var.get(),"Plate width",0.1); py=parse_float(self.plate_height_y_var.get(),"Plate height",0.1)
        t_mm=self._float_or_zero_for_full_same(self.heatsink_fin_thickness_var.get(),parse_float(self.plate_thickness_var.get(),"Plate thickness",0.1))
        across=px if self._heatsink_orientation_key()=="run_y" else py; half=t_mm/20.0
        return -across/2.0+half, across/2.0-half

    def _parse_fin_list(self, raw, fallback, count):
        raw=str(raw).strip().lower()
        if raw in ("","same","even","full"): return [fallback]*count
        vals=[parse_float(p.strip(),"Fin list value") for p in raw.replace(";",",").split(",") if p.strip()]
        if not vals: return [fallback]*count
        if len(vals)<count: vals += [vals[-1]]*(count-len(vals))
        return vals[:count]

    def _set_fins_even(self):
        self.heatsink_enabled_var.set(True); self.heatsink_mode_var.set("Geometry builder"); self.heatsink_fin_positions_var.set("even"); self._update_heatsink_label(); self._schedule_preview()

    def _set_fins_near_resistors(self):
        try:
            self.heatsink_enabled_var.set(True); self.heatsink_mode_var.set("Geometry builder")
            count=self._fin_count_int();
            if count<=0: raise ValueError("Fin count must be above 0.")
            lo,hi=self._fin_across_bounds()
            if count==1: positions=[0.0]
            else:
                even=list(np.linspace(lo,hi,count)); axis=[r.center_x_cm if self._heatsink_orientation_key()=="run_y" else r.center_y_cm for r in self.resistors]
                axis=[min(hi,max(lo,v)) for v in axis]; positions=[]
                for v in axis:
                    if len(positions)<count: positions.append(v)
                for v in even:
                    if len(positions)<count: positions.append(float(v))
                positions=sorted(positions[:count]); min_gap=max(0.05,(hi-lo)/max(1,count-1)*0.20)
                for i in range(1,len(positions)):
                    if positions[i]-positions[i-1]<min_gap: positions[i]=min(hi,positions[i-1]+min_gap)
                for i in range(len(positions)-2,-1,-1):
                    if positions[i+1]-positions[i]<min_gap: positions[i]=max(lo,positions[i+1]-min_gap)
            self.heatsink_fin_positions_var.set(", ".join(f"{p:.2f}" for p in positions)); self._update_heatsink_label(); self._schedule_preview()
        except Exception as e: messagebox.showerror("Fin layout",str(e))

    def _set_fins_edge_biased(self):
        try:
            count=self._fin_count_int();
            if count<=0: raise ValueError("Fin count must be above 0.")
            lo,hi=self._fin_across_bounds(); positions=[0.0] if count==1 else [lo+(hi-lo)*(0.5-0.5*math.cos(math.pi*i/(count-1))) for i in range(count)]
            self.heatsink_fin_positions_var.set(", ".join(f"{p:.2f}" for p in positions)); self._update_heatsink_label(); self._schedule_preview()
        except Exception as e: messagebox.showerror("Fin layout",str(e))

    def _open_fin_layout_designer(self):
        try:
            count=self._fin_count_int();
            if count<=0: raise ValueError("Set number of fins first.")
            lo,hi=self._fin_across_bounds(); default_h=parse_float(self.heatsink_fin_height_var.get(),"Default fin height",0.1)
            positions=([0.0] if count==1 else list(np.linspace(lo,hi,count))) if str(self.heatsink_fin_positions_var.get()).strip().lower()=="even" else self._parse_fin_list(self.heatsink_fin_positions_var.get(),0.0,count)
            heights=self._parse_fin_list(self.heatsink_fin_heights_var.get(),default_h,count)
        except Exception as e:
            messagebox.showerror("Fin layout",str(e)); return
        win=tk.Toplevel(self); win.title("Fin layout designer"); win.geometry("560x520"); win.transient(self)
        frame=ttk.Frame(win,padding=10); frame.pack(fill="both",expand=True)
        tree=ttk.Treeview(frame,columns=("idx","pos","height"),show="headings",height=12)
        for c,h,w in [("idx","Fin",50),("pos","Position cm",120),("height","Height mm",120)]: tree.heading(c,text=h); tree.column(c,width=w,anchor="center")
        tree.grid(row=0,column=0,columnspan=4,sticky="nsew"); sb=ttk.Scrollbar(frame,orient="vertical",command=tree.yview); sb.grid(row=0,column=4,sticky="ns"); tree.configure(yscrollcommand=sb.set); frame.rowconfigure(0,weight=1); frame.columnconfigure(2,weight=1)
        pos_var=tk.StringVar(); height_var=tk.StringVar(); ttk.Label(frame,text="Position cm").grid(row=1,column=0,sticky="w",pady=(8,2)); ttk.Entry(frame,textvariable=pos_var,width=12).grid(row=1,column=1,sticky="ew",pady=(8,2)); ttk.Label(frame,text="Height mm").grid(row=1,column=2,sticky="w",pady=(8,2)); ttk.Entry(frame,textvariable=height_var,width=12).grid(row=1,column=3,sticky="ew",pady=(8,2))
        for i,(p,h) in enumerate(zip(positions,heights),1): tree.insert("","end",iid=str(i-1),values=(i,f"{p:.3f}",f"{h:.3f}"))
        def selected():
            sel=tree.selection(); return int(sel[0]) if sel else None
        def on_select(event=None):
            i=selected();
            if i is None: return
            vals=tree.item(str(i),"values"); pos_var.set(vals[1]); height_var.set(vals[2])
        def update_selected():
            i=selected();
            if i is None: messagebox.showinfo("Fin layout","Select a fin first."); return
            p=min(hi,max(lo,parse_float(pos_var.get(),"Position"))); h=parse_float(height_var.get(),"Height",0.1); tree.item(str(i),values=(i+1,f"{p:.3f}",f"{h:.3f}"))
        def set_even():
            vals=[0.0] if count==1 else list(np.linspace(lo,hi,count))
            for i,item in enumerate(tree.get_children()): tree.item(item,values=(i+1,f"{vals[i]:.3f}",tree.item(item,"values")[2]))
        def set_edge():
            vals=[0.0] if count==1 else [lo+(hi-lo)*(0.5-0.5*math.cos(math.pi*i/(count-1))) for i in range(count)]
            for i,item in enumerate(tree.get_children()): tree.item(item,values=(i+1,f"{vals[i]:.3f}",tree.item(item,"values")[2]))
        def set_res():
            self._set_fins_near_resistors(); vals=self._parse_fin_list(self.heatsink_fin_positions_var.get(),0.0,count)
            for i,item in enumerate(tree.get_children()): tree.item(item,values=(i+1,f"{vals[i]:.3f}",tree.item(item,"values")[2]))
        def apply_close(close=False):
            pos=[]; hs=[]
            for item in tree.get_children():
                vals=tree.item(item,"values"); pos.append(min(hi,max(lo,float(vals[1])))); hs.append(max(0.1,float(vals[2])))
            self.heatsink_enabled_var.set(True); self.heatsink_mode_var.set("Geometry builder"); self.heatsink_fin_positions_var.set(", ".join(f"{p:.3f}" for p in pos)); self.heatsink_fin_heights_var.set(", ".join(f"{h:.3f}" for h in hs)); self._update_heatsink_label(); self._schedule_preview()
            if close: win.destroy()
        tree.bind("<<TreeviewSelect>>",on_select)
        if tree.get_children(): tree.selection_set("0"); on_select()
        b=ttk.Frame(frame); b.grid(row=2,column=0,columnspan=4,sticky="ew",pady=(8,0)); ttk.Button(b,text="Update selected",command=update_selected).pack(side="left",padx=(0,4)); ttk.Button(b,text="Even",command=set_even).pack(side="left",padx=(0,4)); ttk.Button(b,text="Edge-biased",command=set_edge).pack(side="left",padx=(0,4)); ttk.Button(b,text="Near resistors",command=set_res).pack(side="left")
        ttk.Label(frame,text=f"Allowed position range: {lo:.2f} to {hi:.2f} cm on the fin spread axis. Positions are centerlines.",foreground="#555",wraplength=500).grid(row=3,column=0,columnspan=4,sticky="w",pady=(10,0))
        c=ttk.Frame(frame); c.grid(row=4,column=0,columnspan=4,sticky="ew",pady=(12,0)); ttk.Button(c,text="Apply",command=lambda:apply_close(False)).pack(side="left",padx=(0,4)); ttk.Button(c,text="Apply and close",command=lambda:apply_close(True)).pack(side="left",padx=(0,4)); ttk.Button(c,text="Close",command=win.destroy).pack(side="right")


    # ------------------------------- status -------------------------------
    def _set_status(self,text):
        self.status_text.configure(state="normal"); self.status_text.delete("1.0","end"); self.status_text.insert("end",text); self.status_text.configure(state="disabled")
    def _append_status(self,text):
        self.status_text.configure(state="normal"); self.status_text.insert("end","\n"+text); self.status_text.see("end"); self.status_text.configure(state="disabled")
    def _set_progress(self,text):
        cur=self.status_text.get("1.0","end").strip(); lines=cur.splitlines() if cur else []
        if lines and lines[-1].startswith("Progress:"): lines[-1]="Progress: "+text
        else: lines.append("Progress: "+text)
        self._set_status("\n".join(lines))


def main():
    app=ThermalPlateGUI(); app.mainloop()

if __name__ == "__main__":
    main()
