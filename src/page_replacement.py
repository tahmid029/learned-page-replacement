from collections import deque

def fifo(trace, capacity):
    frames = []
    queue = deque()
    hits = faults = 0

    for page in trace:
        if page in frames:
            hits += 1
        else:
            faults += 1
            if len(frames) < capacity:
                frames.append(page)
                queue.append(page)
            else:
                victim = queue.popleft()
                frames.remove(victim)
                frames.append(page)
                queue.append(page)

    return hits, faults


def lru(trace, capacity):
    frames = []
    last_used = {}
    hits = faults = 0

    for t, page in enumerate(trace):
        if page in frames:
            hits += 1
        else:
            faults += 1
            if len(frames) < capacity:
                frames.append(page)
            else:
                victim = min(frames, key=lambda p: last_used.get(p, -1))
                frames.remove(victim)
                frames.append(page)
        last_used[page] = t

    return hits, faults


def optimal(trace, capacity):
    frames = []
    hits = faults = 0

    for i, page in enumerate(trace):
        if page in frames:
            hits += 1
            continue

        faults += 1
        if len(frames) < capacity:
            frames.append(page)
            continue

        future = trace[i + 1:]
        next_use = {}
        for p in frames:
            try:
                next_use[p] = future.index(p)
            except ValueError:
                next_use[p] = float("inf")

        victim = max(frames, key=lambda p: next_use[p])
        frames.remove(victim)
        frames.append(page)

    return hits, faults


def optimal_victim(trace, position, frames):
    """Return the page Belady's Optimal policy would evict."""
    future = trace[position + 1:]
    next_use = {}

    for p in frames:
        try:
            next_use[p] = future.index(p)
        except ValueError:
            next_use[p] = float("inf")

    return max(frames, key=lambda p: next_use[p])
