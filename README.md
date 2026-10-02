# Learning-Augmented Page Replacement

## 1. Project Overview

This project investigates **learning-augmented page replacement** in operating-system memory management.

Traditional page replacement policies such as FIFO and LRU make decisions using fixed rules. However, real workloads may change their access patterns over time. This project therefore compares classical page replacement policies with a lightweight machine-learning-based eviction policy.

The implemented policies are:

* FIFO (First-In, First-Out)
* LRU (Least Recently Used)
* Optimal (Belady's algorithm)
* Learned Policy using a lightweight Decision Tree classifier

The project also introduces a workload shift and compares the performance of all policies before and after the shift.

---

## 2. Project Objectives

The main objectives are:

1. Implement FIFO page replacement.
2. Implement LRU page replacement.
3. Implement Optimal (Belady's) page replacement.
4. Implement a lightweight learned eviction policy.
5. Generate a synthetic page-access workload.
6. Introduce a workload shift halfway through the execution.
7. Measure hit ratio and page-fault count before and after the shift.
8. Compare the behavior of classical and learned policies.

---

## 3. Project Structure

```text
learned_page_replacement_project/
│
├── src/
│   ├── main.py
│   ├── learned_policy.py
│   ├── page_replacement.py
│   └── workload.py
│
├── results/
│   ├── results.csv
│   ├── hit_ratio.png
│   ├── page_faults.png
│   └── phase_comparison.png
│
├── report/
│   └── term_paper.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 4. Algorithms

### FIFO

FIFO removes the page that has been present in memory for the longest time.

### LRU

LRU removes the page that has not been accessed for the longest time.

### Optimal

Optimal, also known as Belady's Optimal algorithm, uses future reference information and removes the page whose next use is farthest in the future.

It is used as a theoretical benchmark for comparison.

### Learned Policy

The learned policy uses a lightweight **Decision Tree classifier**.

The model uses page-related features such as:

* Recency
* Frequency
* Age
* Relative frequency
* Relative recency
* Rank-based features

The model is trained using decisions generated from the Optimal policy and then used to select an eviction candidate during simulation.

---

## 5. Workload and Experimental Setup

The final experiment uses:

| Parameter                |         Value |
| ------------------------ | ------------: |
| Number of memory frames  |             5 |
| Total page accesses      |          1000 |
| First phase              |  500 accesses |
| Second phase             |  500 accesses |
| Number of possible pages |            30 |
| Workload shift           | At access 500 |

The workload contains two phases. The first phase follows a locality-heavy/sequential-style pattern, while the second phase changes to a different random/bursty access pattern.

The memory state is maintained continuously across the workload shift.

---

## 6. Machine Learning Training

The learned component uses a Decision Tree classifier.

For the final experiment:

* Training samples: **136,265**
* Validation samples: **34,675**
* Training accuracy: **79.79%**
* Validation accuracy: **79.08%**
* Overall victim-decision accuracy: **58.67%**
* Before-shift victim accuracy: **55.92%**
* After-shift victim accuracy: **60.82%**

The most influential feature in the final Decision Tree was `age_rank`, with a feature importance of approximately **0.7550**.

---

## 7. Final Experimental Results

| Policy  | Phase        | Hits | Page Faults | Hit Ratio |
| ------- | ------------ | ---: | ----------: | --------: |
| FIFO    | Before Shift |  212 |         288 |     42.4% |
| FIFO    | After Shift  |  181 |         319 |     36.2% |
| LRU     | Before Shift |  223 |         277 |     44.6% |
| LRU     | After Shift  |  196 |         304 |     39.2% |
| Optimal | Before Shift |  343 |         157 |     68.6% |
| Optimal | After Shift  |  306 |         194 |     61.2% |
| Learned | Before Shift |  163 |         337 |     32.6% |
| Learned | After Shift  |  196 |         304 |     39.2% |

---

## 8. Results Analysis

FIFO decreased from a hit ratio of **42.4%** before the workload shift to **36.2%** after the shift.

LRU decreased from **44.6%** to **39.2%**.

Optimal decreased from **68.6%** to **61.2%**.

The learned policy behaved differently. Its hit ratio increased from **32.6%** before the shift to **39.2%** after the shift.

Therefore, in this particular synthetic experiment, the learned policy did not outperform Optimal and had lower performance than LRU before the shift. However, after the workload shift, its hit ratio increased and reached the same hit ratio as LRU.

These results demonstrate that different replacement strategies can respond differently when the workload changes.

---

## 9. Generated Results

The `results/` directory contains:

### `results.csv`

Contains the numerical results for all four policies before and after the workload shift.

### `hit_ratio.png`

Compares hit ratios before and after the workload shift.

### `page_faults.png`

Compares page-fault counts before and after the workload shift.

### `phase_comparison.png`

Shows the change in hit ratio caused by the workload shift.

---

## 10. Requirements

The project requires Python 3 and the following Python packages:

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

The required packages are listed in `requirements.txt`.

---

## 11. Installation

Create and activate a Python virtual environment.

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

## 12. How to Run

From the project root directory:

```powershell
python src\main.py
```

The program will:

1. Generate the workload.
2. Build the training and validation datasets.
3. Train the Decision Tree.
4. Run FIFO.
5. Run LRU.
6. Run Optimal.
7. Run the Learned Policy.
8. Calculate hits, page faults, and hit ratios.
9. Save the final results to the `results/` directory.
10. Generate the comparison charts.

---

## 13. Reproducing the Final Experiment

The final experiment uses:

```text
Frames: 5
Total accesses: 1000
Workload shift: access 500
Pages: 30
Random seed: 42
```

Running:

```powershell
python src\main.py
```

reproduces the experiment using the configuration defined in `src/main.py`.

---

## 14. AI Assistance Disclosure

AI assistance was used during the development of this project.

ChatGPT was used as an implementation and debugging assistant to help explain operating-system concepts, suggest code structure, troubleshoot implementation issues, improve the learned-policy implementation, and assist with documentation.

The student ran the implementation, generated and reviewed the experimental results, and is responsible for understanding and explaining the submitted code, methodology, and results.

---

## 15. Conclusion

This project demonstrates a comparison between classical page replacement algorithms and a lightweight learning-based eviction policy under a changing workload.

The experiment shows that workload changes can affect different page replacement policies differently. The learned policy did not outperform the theoretical Optimal policy in the experiment, but its behavior after the workload shift differed from the classical policies and its hit ratio increased during the second phase.

The project therefore provides an experimental demonstration of how learning-based components can be integrated with traditional operating-system memory-management strategies.
