import random
import copy
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import List, Optional


MAX_HEIGHT = 4

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
    actual_departure_time: float
    service: str = ""
    vessel: str = ""


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
    def __init__(self, n_import_stacks=4, n_export_stacks=4):
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


def generate_fake_containers(n=20, seed=1):
    random.seed(seed)
    containers = []

    for i in range(n):
        bucket = random.choices(
            [0, 1, 2, 3],
            weights=[0.20, 0.27, 0.24, 0.29]
        )[0]

        actual_departure = {
            0: random.uniform(0, 36),
            1: random.uniform(36, 72),
            2: random.uniform(72, 120),
            3: random.uniform(120, 336),
        }[bucket]

        containers.append(
            Container(
                container_id=f"C{i:03d}",
                ied_code=random.choice(["IMPORT", "EXPORT"]),
                size_ft=random.choice([20, 40]),
                weight=random.uniform(5000, 32000),
                pred_bucket=bucket,
                actual_departure_time=actual_departure,
                service=random.choice(["NE2", "FAL1", "MSC_SWAN"]),
                vessel=random.choice(["V1", "V2", "V3"]),
            )
        )

    return containers


class SimState:
    def __init__(self, strategy, containers, n_import_stacks, n_export_stacks):
        self.strategy = strategy
        self.containers = copy.deepcopy(containers)
        self.yard = Yard(n_import_stacks, n_export_stacks)
        self.departure_queue = sorted(
            copy.deepcopy(containers),
            key=lambda c: c.actual_departure_time
        )
        self.current_step = 0
        self.total_reshuffles = 0
        self.placed_count = 0
        self.retrieved_count = 0
        self.failed_placements = 0
        self.last_action = "Ready"

    def step(self):
        if self.current_step < len(self.containers):
            c = self.containers[self.current_step]

            if self.strategy == "bucket":
                placed_stack = self.yard.place_bucket(c)
            else:
                placed_stack = self.yard.place_baseline(c)

            self.current_step += 1
            self.placed_count += 1

            if placed_stack is None:
                self.failed_placements += 1
                self.last_action = f"Failed to place {c.container_id}"
            else:
                self.last_action = (
                    f"Placed {c.container_id} | "
                    f"{c.ied_code} | B{c.pred_bucket} | {placed_stack}"
                )

            return

        if not self.departure_queue:
            self.last_action = "Complete"
            return

        c = self.departure_queue.pop(0)
        stack = self.yard.find_stack(c.container_id)

        if stack is None:
            self.last_action = f"{c.container_id} already gone / not placed"
            return

        target, reshuffled = stack.retrieve_with_reshuffle(c.container_id)
        self.total_reshuffles += len(reshuffled)
        self.retrieved_count += 1

        for r in reshuffled:
            self.yard.place_reshuffled_container(r)

        self.last_action = (
            f"Retrieved {c.container_id} | "
            f"Reshuffled {len(reshuffled)}"
        )

    def is_complete(self):
        return (
            self.current_step >= len(self.containers)
            and len(self.departure_queue) == 0
        )


class YardSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("visual")
        self.root.geometry("1500x820")

        self.containers: List[Container] = []
        self.bucket_sim: Optional[SimState] = None
        self.baseline_sim: Optional[SimState] = None
        self.running = False

        self._build_ui()

    def _build_ui(self):
        self.main = ttk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.left = ttk.Frame(self.main, padding=10)
        self.left.pack(side=tk.LEFT, fill=tk.Y)

        self.right = ttk.Frame(self.main)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.bucket_canvas = tk.Canvas(self.right, bg="white", width=650, height=650)
        self.bucket_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.baseline_canvas = tk.Canvas(self.right, bg="white", width=650, height=650)
        self.baseline_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(
            self.left,
            text="Settings",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", pady=(0, 10))

        self.n_import_var = tk.IntVar(value=4)
        self.n_export_var = tk.IntVar(value=4)
        self.n_container_var = tk.IntVar(value=20)
        self.seed_var = tk.IntVar(value=1)
        self.speed_var = tk.IntVar(value=500)

        self._spin("Import stacks", self.n_import_var, 1, 20)
        self._spin("Export stacks", self.n_export_var, 1, 20)
        self._spin("Containers", self.n_container_var, 5, 200)
        self._spin("Random seed", self.seed_var, 0, 9999)

        ttk.Button(self.left, text="Get new set of containers", command=self.generate).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Start left and right (run)", command=self.start).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Pause", command=self.pause).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Step left and right", command=self.step).pack(fill=tk.X, pady=4)
        ttk.Button(self.left, text="Start over", command=self.reset).pack(fill=tk.X, pady=4)

        # ttk.Label(self.left, text="Speed").pack(anchor="w", pady=(15, 0))
        # ttk.Scale(
        #     self.left,
        #     from_=1000,
        #     to=100,
        #     variable=self.speed_var,
        #     orient=tk.HORIZONTAL
        # ).pack(fill=tk.X)

        ttk.Separator(self.left).pack(fill=tk.X, pady=15)

        self.stats_label = ttk.Label(self.left, text="", justify=tk.LEFT)
        self.stats_label.pack(anchor="w")

        ttk.Separator(self.left).pack(fill=tk.X, pady=15)

        self.action_label = ttk.Label(
            self.left,
            # text="Click Generate Containers",
            wraplength=230,
            justify=tk.LEFT
        )
        self.action_label.pack(anchor="w")

    def _spin(self, label, variable, low, high):
        ttk.Label(self.left, text=label).pack(anchor="w")
        ttk.Spinbox(
            self.left,
            from_=low,
            to=high,
            textvariable=variable,
            width=10
        ).pack(anchor="w", pady=(0, 8))

    def generate(self):
        self.containers = generate_fake_containers(
            n=self.n_container_var.get(),
            seed=self.seed_var.get()
        )

        self.bucket_sim = SimState(
            strategy="bucket",
            containers=self.containers,
            n_import_stacks=self.n_import_var.get(),
            n_export_stacks=self.n_export_var.get()
        )

        self.baseline_sim = SimState(
            strategy="baseline",
            containers=self.containers,
            n_import_stacks=self.n_import_var.get(),
            n_export_stacks=self.n_export_var.get()
        )

        self.running = False
        # self.action_label.config(text="Generated same containers for both simulations.")
        self.draw_all()

    def start(self):
        if self.bucket_sim is None or self.baseline_sim is None:
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
        if self.bucket_sim is None or self.baseline_sim is None:
            self.generate()

        if not self.bucket_sim.is_complete():
            self.bucket_sim.step()

        if not self.baseline_sim.is_complete():
            self.baseline_sim.step()

        self.action_label.config(
            text=(
                "Bucket Heuristic:\n"
                f"{self.bucket_sim.last_action}\n\n"
                "Rules Only:\n"
                f"{self.baseline_sim.last_action}"
            )
        )

        self.draw_all()

        if self.bucket_sim.is_complete() and self.baseline_sim.is_complete():
            self.running = False

    def draw_all(self):
        self.draw_canvas(self.bucket_canvas, self.bucket_sim, "BUCKET HEURISTIC")
        self.draw_canvas(self.baseline_canvas, self.baseline_sim, "RULES ONLY BASELINE")
        self.update_stats()

    def draw_canvas(self, canvas, sim, title):
        canvas.delete("all")
        canvas.create_text(330, 25, text=title, font=("Arial", 15, "bold"))

        self.draw_legend(canvas)

        if sim is None:
            return

        import_stacks = [s for s in sim.yard.stacks if s.yard_type == "IMPORT"]
        export_stacks = [s for s in sim.yard.stacks if s.yard_type == "EXPORT"]

        canvas.create_text(165, 80, text="IMPORT", fill="blue", font=("Arial", 12, "bold"))
        canvas.create_text(495, 80, text="EXPORT", fill="green", font=("Arial", 12, "bold"))

        self.draw_stack_group(canvas, import_stacks, start_x=35, base_y=540)
        self.draw_stack_group(canvas, export_stacks, start_x=360, base_y=540)

        canvas.create_text(
            330,
            620,
            text=(
                f"Reshuffles: {sim.total_reshuffles} | "
                f"Placed: {sim.placed_count}/{len(sim.containers)} | "
                f"Retrieved: {sim.retrieved_count}/{len(sim.containers)} | "
                f"Failed: {sim.failed_placements}"
            ),
            font=("Arial", 10, "bold")
        )

        canvas.create_text(
            330,
            645,
            text=sim.last_action,
            font=("Arial", 9),
            fill="black"
        )

    def draw_legend(self, canvas):
        x = 35
        y = 45

        for bucket in [0, 1, 2, 3]:
            canvas.create_rectangle(
                x, y, x + 16, y + 16,
                fill=BUCKET_COLORS[bucket],
                outline="black"
            )
            canvas.create_text(
                x + 22,
                y + 8,
                text=f"B{bucket}",
                anchor="w",
                font=("Arial", 9)
            )
            x += 70

    def draw_stack_group(self, canvas, stacks, start_x, base_y):
        box_w = 30
        box_h = 28
        gap = 12

        for idx, stack in enumerate(stacks):
            x = start_x + idx * (box_w + gap)

            canvas.create_rectangle(
                x - 4,
                base_y + 4,
                x + box_w + 4,
                base_y + 12,
                fill="#bdbdbd",
                outline="#777"
            )

            canvas.create_text(
                x + box_w / 2,
                base_y + 28,
                text=stack.stack_id,
                font=("Arial", 8, "bold")
            )

            for tier in range(MAX_HEIGHT):
                y_top = base_y - (tier + 1) * box_h

                canvas.create_rectangle(
                    x,
                    y_top,
                    x + box_w,
                    y_top + box_h,
                    outline="#dddddd",
                    dash=(3, 3)
                )

            for tier, c in enumerate(stack.containers):
                y_top = base_y - (tier + 1) * box_h
                color = BUCKET_COLORS[c.pred_bucket]

                canvas.create_rectangle(
                    x,
                    y_top,
                    x + box_w,
                    y_top + box_h,
                    fill=color,
                    outline="black"
                )

                canvas.create_text(
                    x + box_w / 2,
                    y_top + box_h / 2,
                    text=c.container_id,
                    font=("Arial", 7, "bold")
                )

    def update_stats(self):
        if self.bucket_sim is None or self.baseline_sim is None:
            self.stats_label.config(text="")
            return

        improvement = (
            self.baseline_sim.total_reshuffles
            - self.bucket_sim.total_reshuffles
        )

        text = (
            "Comparison left (improved) and right (original)\n"
            # "----------------------\n"
            f"Bucket reshuffles: {self.bucket_sim.total_reshuffles}\n"
            f"Rules-only reshuffles: {self.baseline_sim.total_reshuffles}\n"
            f"Reduction: {improvement}\n\n"
            f"Improved (left) failed count: {self.bucket_sim.failed_placements}\n"
            f"Original (right) failed count: {self.baseline_sim.failed_placements}"
        )

        self.stats_label.config(text=text)


if __name__ == "__main__":
    root = tk.Tk()
    app = YardSimApp(root)
    root.mainloop()