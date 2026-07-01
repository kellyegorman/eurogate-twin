import copy
import random
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
    0: 3,
    1: 2,
    2: 1,
    3: 0,
}

BUCKET_COLORS = {
    0: "#e53935",
    1: "#fb8c00",
    2: "#fdd835",
    3: "#43a047",
}


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


class Stack:
    def __init__(self, stack_id, yard_type):
        self.stack_id = stack_id
        self.yard_type = yard_type
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

        if c.weight > below.weight:
            return False

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
        self.stacks = []

        for i in range(n_import_stacks):
            self.stacks.append(Stack(f"IMP_{i}", "IMPORT"))

        for i in range(n_export_stacks):
            self.stacks.append(Stack(f"EXP_{i}", "EXPORT"))

    def feasible_stacks(self, c):
        return [s for s in self.stacks if s.can_place(c)]

    def place_baseline(self, c):
        feasible = self.feasible_stacks(c)

        if not feasible:
            return None

        best = min(feasible, key=lambda s: s.height())
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
                abs(s.height() - preferred_tier),
                s.height()
            )
        )

        best.place(c)
        return best.stack_id

    def place_reshuffled_container(self, c):
        feasible = self.feasible_stacks(c)

        if not feasible:
            return None

        best = min(feasible, key=lambda s: s.height())
        best.place(c)
        return best.stack_id

    def find_stack(self, container_id):
        for s in self.stacks:
            if s.contains(container_id):
                return s
        return None


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


def load_synthetic_containers(n=100, seed=10):
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
                type_code=infer_type_code(row)
            )
        )

    containers.sort(key=lambda c: c.arrival_time)
    return containers


def infer_type_code(row):
    for col in row.index:
        if col.startswith("type_") and row[col] == 1:
            return col.replace("type_", "")
    return "UNK"


class SimState:
    def __init__(self, strategy, containers, n_import_stacks, n_export_stacks):
        self.strategy = strategy
        self.containers = copy.deepcopy(containers)
        self.yard = Yard(n_import_stacks, n_export_stacks)

        self.pending_arrivals = sorted(
            copy.deepcopy(containers),
            key=lambda c: c.arrival_time
        )

        self.pending_departures = sorted(
            copy.deepcopy(containers),
            key=lambda c: c.departure_time
        )

        self.total_reshuffles = 0
        self.placed_count = 0
        self.retrieved_count = 0
        self.failed_placements = 0
        self.hour_events = []
        self.last_action = "Ready"

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
                placed_stack = self.yard.place_bucket(c)
            else:
                placed_stack = self.yard.place_baseline(c)

            self.placed_count += 1

            if placed_stack is None:
                self.failed_placements += 1
                msg = f"{c.arrival_time.strftime('%H:%M:%S')} FAILED place {c.container_id}"
            else:
                msg = f"{c.arrival_time.strftime('%H:%M:%S')} placed {c.container_id} at {placed_stack}"

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


class YardSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time-Based Split-Screen Yard Simulation")
        self.root.geometry("1650x900")

        self.containers = []
        self.bucket_sim = None
        self.baseline_sim = None

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

        self.bucket_canvas = tk.Canvas(self.right, bg="white", width=700, height=620)
        self.bucket_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.baseline_canvas = tk.Canvas(self.right, bg="white", width=700, height=620)
        self.baseline_canvas.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.bucket_text = tk.Text(self.right, height=10, width=80)
        self.bucket_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.baseline_text = tk.Text(self.right, height=10, width=80)
        self.baseline_text.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_columnconfigure(1, weight=1)

        ttk.Label(self.left, text="Settings", font=("Arial", 12, "bold")).pack(anchor="w")

        self.n_import_var = tk.IntVar(value=5)
        self.n_export_var = tk.IntVar(value=5)
        self.n_container_var = tk.IntVar(value=50)
        self.seed_var = tk.IntVar(value=10)
        self.speed_var = tk.IntVar(value=700)

        self._spin("Import stacks", self.n_import_var, 1, 20)
        self._spin("Export stacks", self.n_export_var, 1, 20)
        self._spin("Containers", self.n_container_var, 5, 500)
        self._spin("Random seed", self.seed_var, 0, 9999)

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
        #     orient=tk.HORIZONTAL
        # ).pack(fill=tk.X)

        ttk.Separator(self.left).pack(fill=tk.X, pady=15)

        self.clock_label = ttk.Label(self.left, text="", font=("Arial", 13, "bold"))
        self.clock_label.pack(anchor="w")

        ttk.Separator(self.left).pack(fill=tk.X, pady=15)

        self.stats_label = ttk.Label(self.left, text="", justify=tk.LEFT)
        self.stats_label.pack(anchor="w")

    def _spin(self, label, variable, low, high):
        ttk.Label(self.left, text=label).pack(anchor="w")
        ttk.Spinbox(self.left, from_=low, to=high, textvariable=variable, width=10).pack(anchor="w", pady=(0, 8))

    def generate(self):
        self.containers = load_synthetic_containers(
            n=self.n_container_var.get(),
            seed=self.seed_var.get()
        )

        start = min(c.arrival_time for c in self.containers)
        self.current_time = start.replace(minute=0, second=0, microsecond=0)

        self.bucket_sim = SimState(
            "bucket",
            self.containers,
            self.n_import_var.get(),
            self.n_export_var.get()
        )

        self.baseline_sim = SimState(
            "baseline",
            self.containers,
            self.n_import_var.get(),
            self.n_export_var.get()
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
        canvas.create_text(350, 25, text=title, font=("Arial", 15, "bold"))
        self.draw_legend(canvas)

        if sim is None:
            return

        import_stacks = [s for s in sim.yard.stacks if s.yard_type == "IMPORT"]
        export_stacks = [s for s in sim.yard.stacks if s.yard_type == "EXPORT"]

        canvas.create_text(175, 80, text="IMPORT", fill="blue", font=("Arial", 12, "bold"))
        canvas.create_text(525, 80, text="EXPORT", fill="green", font=("Arial", 12, "bold"))

        self.draw_stack_group(canvas, import_stacks, start_x=40, base_y=540)
        self.draw_stack_group(canvas, export_stacks, start_x=390, base_y=540)

    def draw_legend(self, canvas):
        x = 35
        y = 45
        for bucket in [0, 1, 2, 3]:
            canvas.create_rectangle(x, y, x + 16, y + 16, fill=BUCKET_COLORS[bucket], outline="black")
            canvas.create_text(x + 22, y + 8, text=f"B{bucket}", anchor="w", font=("Arial", 9))
            x += 75

    def draw_stack_group(self, canvas, stacks, start_x, base_y):
        box_w = 48
        box_h = 42
        gap = 14

        for idx, stack in enumerate(stacks):
            x = start_x + idx * (box_w + gap)

            canvas.create_rectangle(x - 4, base_y + 4, x + box_w + 4, base_y + 12, fill="#bdbdbd", outline="#777")
            canvas.create_text(x + box_w / 2, base_y + 28, text=stack.stack_id, font=("Arial", 8, "bold"))

            for tier in range(MAX_HEIGHT):
                y_top = base_y - (tier + 1) * box_h
                canvas.create_rectangle(x, y_top, x + box_w, y_top + box_h, outline="#dddddd", dash=(3, 3))

            for tier, c in enumerate(stack.containers):
                y_top = base_y - (tier + 1) * box_h
                canvas.create_rectangle(
                    x,
                    y_top,
                    x + box_w,
                    y_top + box_h,
                    fill=BUCKET_COLORS[c.pred_bucket],
                    outline="black"
                )
                canvas.create_text(
                    x + box_w / 2,
                    y_top + box_h / 2,
                    text=c.container_id[-4:],
                    font=("Arial", 7, "bold")
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
                f"pred bucket: {c.pred_bucket}\n"
            )

        self.baseline_text.insert(tk.END, "Baseline events this hour:\n")
        for e in self.baseline_sim.hour_events:
            self.baseline_text.insert(tk.END, e + "\n")

        self.baseline_text.insert(tk.END, "\nRecent baseline-side containers:\n")
        for c in self.containers[:10]:
            self.baseline_text.insert(
                tk.END,
                f"{c.container_id}, TEUs: {c.teu}, Weight: {c.weight:.0f}, "
                f"IED: {c.ied_code[0]}, Type: {c.type_code}\n"
            )

    def update_stats(self):
        if self.bucket_sim is None:
            return

        self.clock_label.config(
            text=self.current_time.strftime("%H:%M:%S  %d:%m:%Y")
        )

        reduction = self.baseline_sim.total_reshuffles - self.bucket_sim.total_reshuffles

        text = (
            "Bucket\n"
            f"  reshuffles: {self.bucket_sim.total_reshuffles}\n"
            f"  placed: {self.bucket_sim.placed_count}\n"
            f"  retrieved: {self.bucket_sim.retrieved_count}\n"
            f"  failed: {self.bucket_sim.failed_placements}\n\n"
            "Baseline\n"
            f"  reshuffles: {self.baseline_sim.total_reshuffles}\n"
            f"  placed: {self.baseline_sim.placed_count}\n"
            f"  retrieved: {self.baseline_sim.retrieved_count}\n"
            f"  failed: {self.baseline_sim.failed_placements}\n\n"
            f"Reshuffle reduction: {reduction}"
        )

        self.stats_label.config(text=text)


if __name__ == "__main__":
    root = tk.Tk()
    app = YardSimApp(root)
    root.mainloop()