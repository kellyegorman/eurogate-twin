import copy
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd


MAX_HEIGHT = 4

MODEL_PATH = "model_artifacts/dwell_bucket_extra_trees.joblib"
FEATURE_COLS_PATH = "model_artifacts/dwell_bucket_feature_columns.joblib"
MEDIANS_PATH = "model_artifacts/dwell_bucket_feature_medians.joblib"

SYNTHETIC_CSV_PATH = "/Users/kellyg/eurogate-twin-1/synthetic_data/synthetic_1000_rows.csv"  

PREFERRED_TIER_FROM_BUCKET = {
    0: 3,  # top
    1: 2,
    2: 1,
    3: 0,  # bottom
}

BUCKET_COLORS = {
    0: "#e53935",
    1: "#fb8c00",
    2: "#fdd835",
    3: "#43a047",
}

OVERFLOW_PENALTY_PER_CONTAINER = 10
OVERFLOW_PENALTY_PER_YARD = 100

# DATA CLASSES
@dataclass
class Container:
    container_id: str
    ied_code: str
    size_ft: int
    weight: float
    pred_bucket: int
    actual_dwell_hours: float
    predicted_bucket_probs: List[float]
    arrival_time: datetime
    departure_time: datetime
    service: str = ""
    vessel: str = ""
    teu: int = 1
    type_code: str = ""
    is_initial_yard_container: bool = False


class Stack:
    def __init__(self, stack_id, yard_type, yard_zone="MAIN"):
        self.stack_id = stack_id
        self.yard_type = yard_type
        self.yard_zone = yard_zone
        self.containers: List[Container] = []

    def height(self):
        return len(self.containers)

    def can_place(self, c: Container):
        if self.yard_type != c.ied_code:
            return False

        if self.height() >= MAX_HEIGHT:
            return False

        if self.height() == 0:
            return True

        below = self.containers[-1]

        # Weight decreases upward
        if c.weight > below.weight:
            return False

        # Simple size rule
        if c.size_ft == 40 and below.size_ft == 20:
            return False

        return True

    def place(self, c: Container):
        self.containers.append(c)

    def contains(self, container_id):
        return any(c.container_id == container_id for c in self.containers)

    def retrieve_with_reshuffle(self, container_id):
        ids = [c.container_id for c in self.containers]

        if container_id not in ids:
            return None, []

        idx = ids.index(container_id)
        target = self.containers[idx]
        above = self.containers[idx + 1:]

        self.containers = self.containers[:idx]

        return target, above


class Yard:
    def __init__(self, n_import_stacks=5, n_export_stacks=5):
        self.base_import_stacks = n_import_stacks
        self.base_export_stacks = n_export_stacks

        self.stacks: List[Stack] = []
        self.overflow_yard_count = 0
        self.overflow_container_count = 0

        self._add_yard_zone("MAIN", n_import_stacks, n_export_stacks)

    def _add_yard_zone(self, zone_name, n_import_stacks, n_export_stacks):
        for i in range(n_import_stacks):
            self.stacks.append(
                Stack(
                    stack_id=f"{zone_name}_IMP_{i}",
                    yard_type="IMPORT",
                    yard_zone=zone_name,
                )
            )

        for i in range(n_export_stacks):
            self.stacks.append(
                Stack(
                    stack_id=f"{zone_name}_EXP_{i}",
                    yard_type="EXPORT",
                    yard_zone=zone_name,
                )
            )

    def add_overflow_yard(self):
        self.overflow_yard_count += 1
        zone_name = f"OVERFLOW_{self.overflow_yard_count}"

        self._add_yard_zone(
            zone_name,
            self.base_import_stacks,
            self.base_export_stacks,
        )

    def feasible_stacks(self, c, include_overflow=True):
        stacks = self.stacks

        if not include_overflow:
            stacks = [s for s in self.stacks if s.yard_zone == "MAIN"]

        return [s for s in stacks if s.can_place(c)]

    def place_with_overflow(self, c, placement_function):
        stack_id = placement_function(c)

        if stack_id is not None:
            return stack_id, False

        # Main yard failed, so try existing overflow yards or create new ones.
        while True:
            self.add_overflow_yard()
            stack_id = placement_function(c)

            if stack_id is not None:
                self.overflow_container_count += 1
                return stack_id, True

    def place_baseline(self, c):
        feasible = self.feasible_stacks(c)

        if not feasible:
            return None

        best = min(
            feasible,
            key=lambda s: (
                s.yard_zone != "MAIN",
                s.height(),
            )
        )

        best.place(c)
        return best.stack_id

    def place_bucket(self, c):
        feasible = self.feasible_stacks(c)

        if not feasible:
            return None

        preferred_tier = PREFERRED_TIER_FROM_BUCKET[c.pred_bucket]

        best = min(
            feasible,
            key=lambda s: (
                s.yard_zone != "MAIN",
                abs(s.height() - preferred_tier),
                s.height(),
            )
        )

        best.place(c)
        return best.stack_id

    def place_reshuffled_container(self, c):
        feasible = self.feasible_stacks(c)

        if not feasible:
            self.add_overflow_yard()
            feasible = self.feasible_stacks(c)

        best = min(
            feasible,
            key=lambda s: (
                s.yard_zone != "MAIN",
                s.height(),
            )
        )

        best.place(c)

        if best.yard_zone != "MAIN":
            self.overflow_container_count += 1

        return best.stack_id

    def find_stack(self, container_id):
        for s in self.stacks:
            if s.contains(container_id):
                return s
        return None

    def overflow_penalty(self):
        return (
            self.overflow_container_count * OVERFLOW_PENALTY_PER_CONTAINER
            + self.overflow_yard_count * OVERFLOW_PENALTY_PER_YARD
        )

    def main_stacks(self):
        return [s for s in self.stacks if s.yard_zone == "MAIN"]

    def overflow_stacks(self):
        return [s for s in self.stacks if s.yard_zone != "MAIN"]


# MODEL + DATA LOADING
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    medians = joblib.load(MEDIANS_PATH)
    return model, feature_cols, medians


def prepare_features_for_model(df, feature_cols, medians):
    X = df.copy()

    drop_cols = [
        "dwell_hours",
        "dwell_bucket",
        "dwell_bucket_candidate",
        "dwell_bucket_3",
        "containerId",
        "snapshot_time",
        "snapshot_date",
        "snapshot_year",
        "snapshot_hour",
        "snapshot_minute",
        "snapshot_second",
        "arrival_year",
        "arrivalLocationX",
        "arrivalLocationY",
        "arrivalLocationZ",
        "raw_arrivalPolCode",
        "raw_arrivalVesselName",
        "raw_arrivalServiceName",
        "raw_arrivalServiceCode",
        "occupied_slot_count",
        "occupied_bay_count",
        "avg_stack_height",
        "max_stack_height",
        "containers_per_occupied_slot",
        "containers_per_occupied_bay",
    ]

    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    X = X.drop(columns=X.select_dtypes(exclude=[np.number]).columns, errors="ignore")

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians)
    X = X.astype(np.float32)

    return X


def infer_type_code(row):
    for col in row.index:
        if col.startswith("type_") and row[col] == 1:
            return col.replace("type_", "")
    return "UNK"


def load_synthetic_containers(n=100, seed=42):
    model, feature_cols, medians = load_model_artifacts()

    df = pd.read_csv(SYNTHETIC_CSV_PATH)

    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = df.head(n).copy()

    X = prepare_features_for_model(df, feature_cols, medians)

    pred_bucket = model.predict(X)
    pred_prob = model.predict_proba(X)

    base_time = datetime(2024, 11, 6, 0, 0, 0)
    rng = np.random.default_rng(seed)

    containers = []

    for i, row in df.reset_index(drop=True).iterrows():
        arrival_offset = float(rng.uniform(0, 12))
        arrival_time = base_time + timedelta(hours=arrival_offset)

        actual_dwell_hours = float(row.get("dwell_hours", rng.uniform(8, 200)))
        departure_time = arrival_time + timedelta(hours=actual_dwell_hours)

        ied_val = int(row.get("iedCode", 0))
        ied_code = "EXPORT" if ied_val == 1 else "IMPORT"

        teu = int(row.get("teu", 1))
        size_ft = 40 if teu >= 2 else 20

        containers.append(
            Container(
                container_id=str(row.get("containerId", f"C{i:03d}")),
                ied_code=ied_code,
                size_ft=size_ft,
                weight=float(row.get("gross", 15000)),
                pred_bucket=int(pred_bucket[i]),
                actual_dwell_hours=actual_dwell_hours,
                predicted_bucket_probs=list(pred_prob[i]),
                arrival_time=arrival_time,
                departure_time=departure_time,
                service=str(row.get("raw_arrivalServiceName", "synthetic_service")),
                vessel=str(row.get("raw_arrivalVesselName", "synthetic_vessel")),
                teu=teu,
                type_code=infer_type_code(row),
            )
        )

    containers.sort(key=lambda c: c.arrival_time)
    return containers


def generate_initial_yard_containers(n=50, seed=42, base_time=None):
    rng = np.random.default_rng(seed)

    if base_time is None:
        base_time = datetime(2024, 11, 6, 0, 0, 0)

    containers = []

    for i in range(n):
        bucket = int(rng.choice([0, 1, 2, 3], p=[0.20, 0.27, 0.24, 0.29]))

        dwell_remaining = {
            0: rng.uniform(2, 24),
            1: rng.uniform(12, 48),
            2: rng.uniform(36, 96),
            3: rng.uniform(72, 240),
        }[bucket]

        arrival_time = base_time - timedelta(hours=float(rng.uniform(6, 168)))
        departure_time = base_time + timedelta(hours=float(dwell_remaining))

        teu = int(rng.choice([1, 2], p=[0.55, 0.45]))

        containers.append(
            Container(
                container_id=f"OLD_{i:03d}",
                ied_code=str(rng.choice(["IMPORT", "EXPORT"])),
                size_ft=40 if teu == 2 else 20,
                weight=float(rng.uniform(5000, 32000)),
                pred_bucket=bucket,
                actual_dwell_hours=float((departure_time - arrival_time).total_seconds() / 3600),
                predicted_bucket_probs=[0.25, 0.25, 0.25, 0.25],
                arrival_time=arrival_time,
                departure_time=departure_time,
                service=str(rng.choice(["NE2", "FAL1", "MSC_SWAN"])),
                vessel=str(rng.choice(["V1", "V2", "V3"])),
                teu=teu,
                type_code=str(rng.choice(["DC", "RF", "TK"])),
                is_initial_yard_container=True,
            )
        )

    return containers


# SIMULATION STATE
class SimState:
    def __init__(
        self,
        strategy,
        containers,
        n_import_stacks,
        n_export_stacks,
        initial_containers=None,
    ):
        self.strategy = strategy
        self.containers = copy.deepcopy(containers)
        self.initial_containers = copy.deepcopy(initial_containers or [])

        self.yard = Yard(n_import_stacks, n_export_stacks)

        self.pending_arrivals = sorted(
            copy.deepcopy(containers),
            key=lambda c: c.arrival_time,
        )

        self.pending_departures = sorted(
            copy.deepcopy(containers + self.initial_containers),
            key=lambda c: c.departure_time,
        )

        self.total_reshuffles = 0
        self.placed_count = 0
        self.initial_placed_count = 0
        self.retrieved_count = 0
        self.hour_events = []
        self.last_action = "Ready"

        self.populate_initial_yard()

    def populate_initial_yard(self):
        for c in self.initial_containers:
            if self.strategy == "bucket":
                stack_id, used_overflow = self.yard.place_with_overflow(c, self.yard.place_bucket)
            else:
                stack_id, used_overflow = self.yard.place_with_overflow(c, self.yard.place_baseline)

            self.initial_placed_count += 1

    def process_hour(self, current_time, next_time):
        self.hour_events = []

        arrivals_now = [
            c for c in self.pending_arrivals
            if current_time <= c.arrival_time < next_time
        ]

        self.pending_arrivals = [
            c for c in self.pending_arrivals
            if c.arrival_time >= next_time
        ]

        for c in arrivals_now:
            if self.strategy == "bucket":
                stack_id, used_overflow = self.yard.place_with_overflow(c, self.yard.place_bucket)
            else:
                stack_id, used_overflow = self.yard.place_with_overflow(c, self.yard.place_baseline)

            self.placed_count += 1

            overflow_text = " OVERFLOW" if used_overflow else ""
            msg = f"{c.arrival_time.strftime('%H:%M:%S')} placed {c.container_id} at {stack_id}{overflow_text}"

            self.hour_events.append(msg)
            self.last_action = msg

        departures_now = [
            c for c in self.pending_departures
            if current_time <= c.departure_time < next_time
        ]

        self.pending_departures = [
            c for c in self.pending_departures
            if c.departure_time >= next_time
        ]

        for c in departures_now:
            stack = self.yard.find_stack(c.container_id)

            if stack is None:
                msg = f"{c.departure_time.strftime('%H:%M:%S')} skip retrieve {c.container_id}"
                self.hour_events.append(msg)
                self.last_action = msg
                continue

            target, reshuffled = stack.retrieve_with_reshuffle(c.container_id)

            self.total_reshuffles += len(reshuffled)
            self.retrieved_count += 1

            for r in reshuffled:
                self.yard.place_reshuffled_container(r)

            msg = (
                f"{c.departure_time.strftime('%H:%M:%S')} retrieved {c.container_id}; "
                f"reshuffled {len(reshuffled)}"
            )

            self.hour_events.append(msg)
            self.last_action = msg

    def is_complete(self):
        return not self.pending_arrivals and not self.pending_departures

    def overflow_penalty(self):
        return self.yard.overflow_penalty()

    def total_cost(self):
        return self.total_reshuffles + self.overflow_penalty()


# UI
class YardSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time-Based Split-Screen Yard Simulation with Overflow")
        self.root.geometry("1700x920")

        self.containers = []
        self.initial_containers = []

        self.bucket_sim: Optional[SimState] = None
        self.baseline_sim: Optional[SimState] = None

        self.running = False
        self.current_time = datetime(2024, 11, 6, 0, 0, 0)
        self.time_step_hours = 1

        self._build_ui()

    def _build_ui(self):
        self.main = ttk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.left = ttk.Frame(self.main, padding=10)
        self.left.pack(side=tk.LEFT, fill=tk.Y)

        self.right = ttk.Frame(self.main)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.bucket_canvas = tk.Canvas(self.right, bg="white", width=720, height=620)
        self.bucket_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.baseline_canvas = tk.Canvas(self.right, bg="white", width=720, height=620)
        self.baseline_canvas.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.bucket_text = tk.Text(self.right, height=11, width=85)
        self.bucket_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.baseline_text = tk.Text(self.right, height=11, width=85)
        self.baseline_text.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_columnconfigure(1, weight=1)

        ttk.Label(self.left, text="Settings", font=("Arial", 12, "bold")).pack(anchor="w")

        self.n_import_var = tk.IntVar(value=5)
        self.n_export_var = tk.IntVar(value=5)
        self.n_container_var = tk.IntVar(value=100)
        self.seed_var = tk.IntVar(value=42)
        self.speed_var = tk.IntVar(value=700)

        self.populate_initial_var = tk.BooleanVar(value=True)
        self.initial_container_var = tk.IntVar(value=60)

        self._spin("Import stacks", self.n_import_var, 1, 20)
        self._spin("Export stacks", self.n_export_var, 1, 20)
        self._spin("New containers", self.n_container_var, 5, 500)
        self._spin("Random seed", self.seed_var, 0, 9999)

        ttk.Checkbutton(
            self.left,
            text="Populate yard in advance",
            variable=self.populate_initial_var,
        ).pack(anchor="w", pady=(8, 2))

        self._spin("Initial old containers", self.initial_container_var, 0, 500)

        ttk.Button(self.left, text="Load Synthetic + Predict", command=self.generate).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Start", command=self.start).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Pause", command=self.pause).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Step 1 Hour", command=self.step).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)

        # ttk.Label(self.left, text="Animation speed").pack(anchor="w", pady=(15, 0))
        # ttk.Scale(
        #     self.left,
        #     from_=1500,
        #     to=100,
        #     variable=self.speed_var,
        #     orient=tk.HORIZONTAL,
        # ).pack(fill=tk.X)

        ttk.Separator(self.left).pack(fill=tk.X, pady=15)

        self.clock_label = ttk.Label(self.left, text="", font=("Arial", 13, "bold"))
        self.clock_label.pack(anchor="w")

        ttk.Separator(self.left).pack(fill=tk.X, pady=15)

        self.stats_label = ttk.Label(self.left, text="", justify=tk.LEFT)
        self.stats_label.pack(anchor="w")

    def _spin(self, label, variable, low, high):
        ttk.Label(self.left, text=label).pack(anchor="w")
        ttk.Spinbox(
            self.left,
            from_=low,
            to=high,
            textvariable=variable,
            width=10,
        ).pack(anchor="w", pady=(0, 8))

    def generate(self):
        self.containers = load_synthetic_containers(
            n=self.n_container_var.get(),
            seed=self.seed_var.get(),
        )

        start = min(c.arrival_time for c in self.containers)
        self.current_time = start.replace(minute=0, second=0, microsecond=0)

        if self.populate_initial_var.get():
            self.initial_containers = generate_initial_yard_containers(
                n=self.initial_container_var.get(),
                seed=self.seed_var.get() + 999,
                base_time=self.current_time,
            )
        else:
            self.initial_containers = []

        self.bucket_sim = SimState(
            "bucket",
            self.containers,
            self.n_import_var.get(),
            self.n_export_var.get(),
            initial_containers=self.initial_containers,
        )

        self.baseline_sim = SimState(
            "baseline",
            self.containers,
            self.n_import_var.get(),
            self.n_export_var.get(),
            initial_containers=self.initial_containers,
        )

        self.running = False
        self.draw_all()

    def start(self):
        if self.bucket_sim is None:
            self.generate()

        self.running = True
        self.run_loop()

    def pause(self):
        self.running = False

    def reset(self):
        self.generate()

    def run_loop(self):
        if self.running:
            self.step()
            self.root.after(self.speed_var.get(), self.run_loop)

    def step(self):
        if self.bucket_sim is None:
            self.generate()

        next_time = self.current_time + timedelta(hours=self.time_step_hours)

        self.bucket_sim.process_hour(self.current_time, next_time)
        self.baseline_sim.process_hour(self.current_time, next_time)

        self.current_time = next_time
        self.draw_all()

        if self.bucket_sim.is_complete() and self.baseline_sim.is_complete():
            self.running = False

    def draw_all(self):
        self.draw_canvas(self.bucket_canvas, self.bucket_sim, "BUCKET HEURISTIC")
        self.draw_canvas(self.baseline_canvas, self.baseline_sim, "RULES-ONLY BASELINE")
        self.update_text_panels()
        self.update_stats()

    def draw_canvas(self, canvas, sim, title):
        canvas.delete("all")
        canvas.create_text(360, 25, text=title, font=("Arial", 15, "bold"))
        self.draw_legend(canvas)

        if sim is None:
            return

        main_import = [s for s in sim.yard.main_stacks() if s.yard_type == "IMPORT"]
        main_export = [s for s in sim.yard.main_stacks() if s.yard_type == "EXPORT"]

        canvas.create_text(180, 80, text="MAIN IMPORT", fill="blue", font=("Arial", 11, "bold"))
        canvas.create_text(535, 80, text="MAIN EXPORT", fill="green", font=("Arial", 11, "bold"))

        self.draw_stack_group(canvas, main_import, start_x=40, base_y=400)
        self.draw_stack_group(canvas, main_export, start_x=390, base_y=400)

        overflow_stacks = sim.yard.overflow_stacks()

        if overflow_stacks:
            canvas.create_text(360, 455, text="OVERFLOW YARD(S)", fill="red", font=("Arial", 11, "bold"))
            shown = overflow_stacks[:10]
            self.draw_stack_group(canvas, shown, start_x=55, base_y=590, small=True)

            if len(overflow_stacks) > 10:
                canvas.create_text(
                    600,
                    575,
                    text=f"+ {len(overflow_stacks) - 10} more overflow stacks",
                    font=("Arial", 9),
                    fill="red",
                )

    def draw_legend(self, canvas):
        x = 35
        y = 45

        for bucket in [0, 1, 2, 3]:
            canvas.create_rectangle(
                x,
                y,
                x + 16,
                y + 16,
                fill=BUCKET_COLORS[bucket],
                outline="black",
            )
            canvas.create_text(
                x + 22,
                y + 8,
                text=f"B{bucket}",
                anchor="w",
                font=("Arial", 9),
            )
            x += 75

    def draw_stack_group(self, canvas, stacks, start_x, base_y, small=False):
        box_w = 48 if not small else 42
        box_h = 42 if not small else 32
        gap = 14 if not small else 10

        for idx, stack in enumerate(stacks):
            x = start_x + idx * (box_w + gap)

            canvas.create_rectangle(
                x - 4,
                base_y + 4,
                x + box_w + 4,
                base_y + 12,
                fill="#bdbdbd",
                outline="#777",
            )

            canvas.create_text(
                x + box_w / 2,
                base_y + 28,
                text=stack.stack_id.replace("MAIN_", ""),
                font=("Arial", 7, "bold"),
            )

            for tier in range(MAX_HEIGHT):
                y_top = base_y - (tier + 1) * box_h
                canvas.create_rectangle(
                    x,
                    y_top,
                    x + box_w,
                    y_top + box_h,
                    outline="#dddddd",
                    dash=(3, 3),
                )

            for tier, c in enumerate(stack.containers):
                y_top = base_y - (tier + 1) * box_h

                outline = "black"
                width = 1

                if c.is_initial_yard_container:
                    outline = "#283593"
                    width = 3

                canvas.create_rectangle(
                    x,
                    y_top,
                    x + box_w,
                    y_top + box_h,
                    fill=BUCKET_COLORS[c.pred_bucket],
                    outline=outline,
                    width=width,
                )

                canvas.create_text(
                    x + box_w / 2,
                    y_top + box_h / 2,
                    text=c.container_id[-4:],
                    font=("Arial", 7, "bold"),
                )

    def update_text_panels(self):
        self.bucket_text.delete("1.0", tk.END)
        self.baseline_text.delete("1.0", tk.END)

        if self.bucket_sim is None:
            return

        self.bucket_text.insert(tk.END, "Bucket heuristic events this hour:\n")
        for e in self.bucket_sim.hour_events:
            self.bucket_text.insert(tk.END, e + "\n")

        self.bucket_text.insert(tk.END, "\nRecent bucket-side containers:\n")
        for c in self.containers[:10]:
            self.bucket_text.insert(
                tk.END,
                f"{c.container_id}, actual dwell: {c.actual_dwell_hours:.1f}h, "
                f"pred bucket: {c.pred_bucket}\n",
            )

        self.baseline_text.insert(tk.END, "Baseline events this hour:\n")
        for e in self.baseline_sim.hour_events:
            self.baseline_text.insert(tk.END, e + "\n")

        self.baseline_text.insert(tk.END, "\nRecent baseline-side containers:\n")
        for c in self.containers[:10]:
            self.baseline_text.insert(
                tk.END,
                f"{c.container_id}, TEUs: {c.teu}, Weight: {c.weight:.0f}, "
                f"IED: {c.ied_code[0]}, Type: {c.type_code}\n",
            )

    def update_stats(self):
        if self.bucket_sim is None:
            return

        self.clock_label.config(
            text=self.current_time.strftime("%H:%M:%S  %d:%m:%Y")
        )

        reshuffle_reduction = (
            self.baseline_sim.total_reshuffles
            - self.bucket_sim.total_reshuffles
        )

        total_cost_reduction = (
            self.baseline_sim.total_cost()
            - self.bucket_sim.total_cost()
        )

        text = (
            "Bucket\n"
            f"  reshuffles: {self.bucket_sim.total_reshuffles}\n"
            f"  placed new: {self.bucket_sim.placed_count}\n"
            f"  placed old: {self.bucket_sim.initial_placed_count}\n"
            f"  retrieved: {self.bucket_sim.retrieved_count}\n"
            f"  overflow yards: {self.bucket_sim.yard.overflow_yard_count}\n"
            f"  overflow containers: {self.bucket_sim.yard.overflow_container_count}\n"
            f"  overflow penalty: {self.bucket_sim.overflow_penalty()}\n"
            f"  total cost: {self.bucket_sim.total_cost()}\n\n"
            "Baseline\n"
            f"  reshuffles: {self.baseline_sim.total_reshuffles}\n"
            f"  placed new: {self.baseline_sim.placed_count}\n"
            f"  placed old: {self.baseline_sim.initial_placed_count}\n"
            f"  retrieved: {self.baseline_sim.retrieved_count}\n"
            f"  overflow yards: {self.baseline_sim.yard.overflow_yard_count}\n"
            f"  overflow containers: {self.baseline_sim.yard.overflow_container_count}\n"
            f"  overflow penalty: {self.baseline_sim.overflow_penalty()}\n"
            f"  total cost: {self.baseline_sim.total_cost()}\n\n"
            f"Reshuffle reduction: {reshuffle_reduction}\n"
            f"Total cost reduction: {total_cost_reduction}"
        )

        self.stats_label.config(text=text)


if __name__ == "__main__":
    root = tk.Tk()
    app = YardSimApp(root)
    root.mainloop()