# Multi-Domain Digital Generalization Testbed 

This project operates as a comprehensive, paper-quality empirical benchmarking framework designed to evaluate the robustness of advanced Machine Learning models encountering Domain Shift. 

We systematically map **three distinct digit datasets** (MNIST, USPS, SVHN) across a fully connected $O(N^2)$ transfer matrix to empirically review **Domain Adaptation algorithms** (CORAL, DANN), **Structural Risk constraints** (IRM), and **Epistemic Uncertainty mapping** (MC Dropout) against pure baselines & upper bounds.

---

## 🏗️ What We Built (Architectural Upgrades)
To convert the framework from a small baseline test into a statistically robust architecture, we:
1. **Implemented Multi-Seed Determinism (`set_seed`)**: Pinned all environments (CUDA, PyTorch, NumPy) to rule out lucky initializations causing artificial performance peaks.
2. **Added Supervised Ceilings**: Added `target_supervised` (Upper-Bound tracking) and `fine_tuning` (few-shot mappings) algorithms to provide an empirical control layer beyond pure `baseline` cross-entropy constraints.
3. **Restructured Tuning Matrices**: Modified `tune.sh` to run metric-isolated ablation loops (e.g., sweeping `dann_gamma` reverses, `irm_warmup_epochs`, and Dropout passing values).
4. **Deployed Entropy Logarithms**: Transitioned MC Dropout variance into formal Mathematical Predictive Entropy ($H$) histograms directly mapped via log functions.
5. **Created Statistical Aggregators**: Created `aggregate_results.py` to auto-scrape output logs across seeds and print $Mean \pm Std Dev$ statistical validation tables.

---

## 🚀 Commands to Run

You can securely trigger global testing workflows sequentially using the local `.venv` encapsulated bash scripts:

### 1. The Hyperparameter Sweep
To isolate penalty bounds, learning gradients, and warmups incrementally:
```bash
bash tune.sh
```
*This isolates specific parameter changes per algorithm without triggering an infinite $N!$ factorial combinatoric blowout.*

### 2. The Statistical Paper Benchmark
To compile multi-seed, full-permutation executions evaluating generalized behavior:
```bash
bash run_experiments.sh
```
*This loops the 6-way source routing path (MNIST <-> SVHN, USPS <-> MNIST...) recursively against SEEDS 42, 123, and 2026.*

### 3. Aggregate System Logs 
To parse the hundreds of resulting executions down into an interpretable metric string:
```bash
python aggregate_results.py
```

---

## 📂 Where Are The Results?

* **Raw Metric Logs**: All raw outputs including analytical `time.time()` epoch costs, and target subset loss metrics generate automatically inside the **`/logs/`** and **`/logs/tuning/`** directories.
* **Visual Artifacts**: All graphic diagram representations, including t-SNE topological mappings, Class-Confusion Matrices, and MC Dropout Epistemic distributions, output securely inside **`/outputs/`**.

---

## 📝 Where Are The Final Summaries?

To save immense calculation parsing logic, we have proactively generated highly structured markdown reports outlining the specific bounds hit by the network parameters:

1. **[tuning_summary.md](./tuning_summary.md)**: Highlights parameter sweeps and displays algorithms sorted by Peak Target Validation boundaries. Good for identifying volatile configurations (e.g., when IRM penalty blocks training bounds completely).
2. **[experiment_summary.md](./experiment_summary.md)**: Stores the explicit Paper-Quality final calculations mapped across multiple random algorithmic seeds tracking *Mean* and *Standard Deviations*.
