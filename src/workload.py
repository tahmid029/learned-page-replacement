import random


def make_trace(
    phase1_len=500,
    phase2_len=500,
    pages=30,
    seed=42
):
    """
    Generate a two-phase workload.

    Phase 1:
        Locality-heavy workload.

    Phase 2:
        Random + bursty workload.

    The same structure is used for the final test trace.
    """

    rng = random.Random(seed)

    # ========================================================
    # PHASE 1 - LOCALITY HEAVY
    # ========================================================

    phase1 = []

    for i in range(phase1_len):

        block_start = (
            (i // 50)
            % max(1, pages - 7)
        )

        if rng.random() < 0.85:
            phase1.append(
                block_start + rng.randint(0, 7)
            )
        else:
            phase1.append(
                rng.randint(0, pages - 1)
            )

    # ========================================================
    # PHASE 2 - RANDOM + BURSTY
    # ========================================================

    phase2 = []

    burst_pages = [
        rng.randint(0, pages - 1)
        for _ in range(8)
    ]

    for i in range(phase2_len):

        if rng.random() < 0.65:
            phase2.append(
                rng.choice(burst_pages)
            )
        else:
            phase2.append(
                rng.randint(0, pages - 1)
            )

    return phase1 + phase2


def make_training_traces(
    n=100,
    length=1000,
    pages=30,
    seed=100
):
    """
    Generate diverse training traces.

    Each training trace contains both:
        - locality-heavy behavior
        - random/bursty behavior

    This gives the learning model exposure to
    multiple workload patterns.
    """

    traces = []

    phase1_len = length // 2
    phase2_len = length - phase1_len

    for k in range(n):

        traces.append(
            make_trace(
                phase1_len=phase1_len,
                phase2_len=phase2_len,
                pages=pages,
                seed=seed + k
            )
        )

    return traces