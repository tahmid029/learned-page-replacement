from collections import deque

import matplotlib.pyplot as plt
import pandas as pd

from page_replacement import optimal_victim
from workload import make_trace, make_training_traces
from learned_policy import LearnedEvictionPolicy


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

CAPACITY = 5
TOTAL_ACCESSES = 1000
PHASE1_LEN = 500
PHASE2_LEN = 500
PAGES = 30

TEST_SEED = 42

TRAINING_TRACES = 100
TRAINING_LENGTH = 1000
TRAINING_SEED = 100


# ============================================================
# FIFO
# ============================================================

def fifo_phases(trace, capacity, split):

    frames = deque()

    before_hits = 0
    before_faults = 0

    after_hits = 0
    after_faults = 0

    for t, page in enumerate(trace):

        after_shift = t >= split

        if page in frames:

            if after_shift:
                after_hits += 1
            else:
                before_hits += 1

            continue

        # Page fault
        if after_shift:
            after_faults += 1
        else:
            before_faults += 1

        if len(frames) < capacity:

            frames.append(page)

        else:

            frames.popleft()
            frames.append(page)

    return (
        before_hits,
        before_faults,
        after_hits,
        after_faults
    )


# ============================================================
# LRU
# ============================================================

def lru_phases(trace, capacity, split):

    frames = []
    last_used = {}

    before_hits = 0
    before_faults = 0

    after_hits = 0
    after_faults = 0

    for t, page in enumerate(trace):

        after_shift = t >= split

        # ----------------------------------------------------
        # PAGE HIT
        # ----------------------------------------------------

        if page in frames:

            if after_shift:
                after_hits += 1
            else:
                before_hits += 1

            last_used[page] = t

            continue

        # ----------------------------------------------------
        # PAGE FAULT
        # ----------------------------------------------------

        if after_shift:
            after_faults += 1
        else:
            before_faults += 1

        # ----------------------------------------------------
        # FREE FRAME
        # ----------------------------------------------------

        if len(frames) < capacity:

            frames.append(page)
            last_used[page] = t

            continue

        # ----------------------------------------------------
        # REMOVE LEAST RECENTLY USED PAGE
        # ----------------------------------------------------

        victim = min(
            frames,
            key=lambda p: last_used[p]
        )

        frames.remove(victim)
        frames.append(page)

        last_used[page] = t

    return (
        before_hits,
        before_faults,
        after_hits,
        after_faults
    )


# ============================================================
# OPTIMAL / BELADY
# ============================================================

def optimal_phases(trace, capacity, split):

    frames = []

    before_hits = 0
    before_faults = 0

    after_hits = 0
    after_faults = 0

    for t, page in enumerate(trace):

        after_shift = t >= split

        # ----------------------------------------------------
        # PAGE HIT
        # ----------------------------------------------------

        if page in frames:

            if after_shift:
                after_hits += 1
            else:
                before_hits += 1

            continue

        # ----------------------------------------------------
        # PAGE FAULT
        # ----------------------------------------------------

        if after_shift:
            after_faults += 1
        else:
            before_faults += 1

        # ----------------------------------------------------
        # FREE FRAME
        # ----------------------------------------------------

        if len(frames) < capacity:

            frames.append(page)

            continue

        # ----------------------------------------------------
        # OPTIMAL VICTIM
        # ----------------------------------------------------

        victim = optimal_victim(
            trace,
            t,
            frames
        )

        frames.remove(victim)
        frames.append(page)

    return (
        before_hits,
        before_faults,
        after_hits,
        after_faults
    )


# ============================================================
# ADD RESULTS
# ============================================================

def add_rows(
    rows,
    policy,
    before_hits,
    before_faults,
    after_hits,
    after_faults
):

    before_total = before_hits + before_faults
    after_total = after_hits + after_faults

    before_hit_ratio = (
        before_hits / before_total
        if before_total > 0
        else 0
    )

    after_hit_ratio = (
        after_hits / after_total
        if after_total > 0
        else 0
    )

    rows.append(
        {
            "Policy": policy,
            "Phase": "Before Shift",
            "Hits": before_hits,
            "Page Faults": before_faults,
            "Hit Ratio": before_hit_ratio
        }
    )

    rows.append(
        {
            "Policy": policy,
            "Phase": "After Shift",
            "Hits": after_hits,
            "Page Faults": after_faults,
            "Hit Ratio": after_hit_ratio
        }
    )


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

def create_results_directory():

    import os

    os.makedirs(
        "results",
        exist_ok=True
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    df.to_csv(
        "results/results.csv",
        index=False
    )


# ============================================================
# PLOT HIT RATIO
# ============================================================

def plot_hit_ratio(df):

    pivot = df.pivot(
        index="Policy",
        columns="Phase",
        values="Hit Ratio"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Hit Ratio Comparison"
    )

    ax.set_xlabel(
        "Page Replacement Policy"
    )

    ax.set_ylabel(
        "Hit Ratio"
    )

    ax.set_ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        "results/hit_ratio.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT PAGE FAULTS
# ============================================================

def plot_page_faults(df):

    pivot = df.pivot(
        index="Policy",
        columns="Phase",
        values="Page Faults"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Page Fault Comparison"
    )

    ax.set_xlabel(
        "Page Replacement Policy"
    )

    ax.set_ylabel(
        "Number of Page Faults"
    )

    plt.tight_layout()

    plt.savefig(
        "results/page_faults.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PHASE COMPARISON
# ============================================================

def plot_phase_comparison(df):

    policies = df["Policy"].unique()

    before_values = []
    after_values = []

    for policy in policies:

        before = df[
            (df["Policy"] == policy)
            & (df["Phase"] == "Before Shift")
        ]["Hit Ratio"].iloc[0]

        after = df[
            (df["Policy"] == policy)
            & (df["Phase"] == "After Shift")
        ]["Hit Ratio"].iloc[0]

        before_values.append(before)
        after_values.append(after)

    x = range(len(policies))

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x,
        before_values,
        marker="o",
        label="Before Shift"
    )

    plt.plot(
        x,
        after_values,
        marker="o",
        label="After Shift"
    )

    plt.xticks(
        list(x),
        policies
    )

    plt.xlabel(
        "Page Replacement Policy"
    )

    plt.ylabel(
        "Hit Ratio"
    )

    plt.title(
        "Performance Before and After Workload Shift"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "results/phase_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run():

    print(
        "============================================================"
    )

    print(
        "LEARNING-AUGMENTED PAGE REPLACEMENT"
    )

    print(
        "============================================================"
    )

    print()

    print(
        "Experiment configuration:"
    )

    print(
        f"Frames: {CAPACITY}"
    )

    print(
        f"Total accesses: {TOTAL_ACCESSES}"
    )

    print(
        f"Workload shift: access {PHASE1_LEN}"
    )

    print(
        f"Pages: {PAGES}"
    )

    print()

    # ========================================================
    # GENERATE TEST WORKLOAD
    # ========================================================

    trace = make_trace(
        phase1_len=PHASE1_LEN,
        phase2_len=PHASE2_LEN,
        pages=PAGES,
        seed=TEST_SEED
    )

    print(
        "Workload generated."
    )

    # ========================================================
    # GENERATE TRAINING WORKLOADS
    # ========================================================

    training_traces = make_training_traces(
        n=TRAINING_TRACES,
        length=TRAINING_LENGTH,
        pages=PAGES,
        seed=TRAINING_SEED
    )

    # ========================================================
    # TRAIN LEARNED POLICY
    # ========================================================

    print()

    print(
        "Training lightweight Decision Tree..."
    )

    model = LearnedEvictionPolicy(
        CAPACITY
    )

    # IMPORTANT:
    # train() now accepts only training_traces.
    model.train(
        training_traces
    )

    # ========================================================
    # VICTIM DECISION DIAGNOSTIC
    # ========================================================

    model.evaluate_victim_accuracy(
        trace,
        PHASE1_LEN
    )

    # ========================================================
    # RUN FIFO
    # ========================================================

    print()

    print(
        "Running FIFO..."
    )

    fifo_result = fifo_phases(
        trace,
        CAPACITY,
        PHASE1_LEN
    )

    # ========================================================
    # RUN LRU
    # ========================================================

    print(
        "Running LRU..."
    )

    lru_result = lru_phases(
        trace,
        CAPACITY,
        PHASE1_LEN
    )

    # ========================================================
    # RUN OPTIMAL
    # ========================================================

    print(
        "Running Optimal..."
    )

    optimal_result = optimal_phases(
        trace,
        CAPACITY,
        PHASE1_LEN
    )

    # ========================================================
    # RUN LEARNED POLICY
    # ========================================================

    print(
        "Running Learned Policy..."
    )

    learned_result = model.simulate_phases(
        trace,
        PHASE1_LEN
    )

    # ========================================================
    # BUILD RESULTS TABLE
    # ========================================================

    rows = []

    add_rows(
        rows,
        "FIFO",
        *fifo_result
    )

    add_rows(
        rows,
        "LRU",
        *lru_result
    )

    add_rows(
        rows,
        "Optimal",
        *optimal_result
    )

    add_rows(
        rows,
        "Learned",
        *learned_result
    )

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        df.to_string(
            index=False
        )
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    create_results_directory()

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_results(
        df
    )

    # ========================================================
    # CREATE GRAPHS
    # ========================================================

    plot_hit_ratio(
        df
    )

    plot_page_faults(
        df
    )

    plot_phase_comparison(
        df
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "Results saved to:"
    )

    import os

    print(
        os.path.abspath(
            "results"
        )
    )

    print()

    print(
        "Generated files:"
    )

    print(
        " - results.csv"
    )

    print(
        " - hit_ratio.png"
    )

    print(
        " - page_faults.png"
    )

    print(
        " - phase_comparison.png"
    )

    print()

    print(
        "Done."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()