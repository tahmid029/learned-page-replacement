# Learning-Augmented Page Replacement: Classical Algorithms Meet Adaptive Prediction

## 1. Problem Framing

This project studies page replacement under a changing workload. The classical policies FIFO, LRU, and Optimal are implemented and compared with a lightweight Decision Tree based policy.

## 2. Implementation Summary

### Classical Algorithms
- FIFO
- LRU
- Optimal (Belady)

### Learned Component

A Decision Tree classifier uses simple runtime features such as recency, access frequency, and page age to predict which resident page is a good eviction candidate.

## 3. Experimental Setup

- Frame capacity: 5
- Total trace length: 1000 accesses
- Phase 1: locality-heavy/sequential
- Phase 2: random/bursty
- The workload changes at access 500.
- Training is performed using separate synthetic locality-heavy traces.

## 4. Results

Insert the actual charts generated in `results/`.

Do not invent numbers. Use the numbers produced by your experiment.

## 5. Analysis

Discuss:
1. Hit ratio before the shift.
2. Hit ratio after the shift.
3. Page faults before and after the shift.
4. How much each policy changed after the shift.
5. Why a workload change can affect recency/frequency-based decisions.
6. Whether the learned policy adapts well to the new pattern.

## 6. Conclusion

Summarize what the experiment demonstrates about combining a classical OS policy with a lightweight learned component.

## 7. AI Assistance Disclosure

AI assistance was used for implementation guidance, code organization, debugging support, and explanation. The workload execution, generated results, and final analysis were checked and finalized by the student.
