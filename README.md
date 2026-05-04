

# 🚀 CETSS: Cost-Efficient Task Scheduling System


## 📌 Overview

**CETSS (Cost-Efficient Task Scheduling System)** is a research-driven framework for scheduling **workflow DAGs on cloud infrastructure** while minimizing:

* 💰 Execution cost
* ⏱️ Makespan
* 📉 Billing inefficiencies

It extends classical schedulers like **HEFT** and **Min-Min** by introducing:

* Billing-aware scheduling (BPS)
* Slack redistribution (USR)
* Resource-aware marginal cost modeling

---

## 🧠 Key Idea

> Instead of just minimizing execution time, CETSS **optimizes cost under real-world cloud billing constraints**.

---

## 🏗️ Architecture

```text
Trace Layer (Workflow DAGs)
        ↓
Infrastructure Layer (VMs, CPU/RAM, bandwidth)
        ↓
Scheduler Layer (HEFT, Min-Min, CETSS)
        ↓
Accounting Layer (Billing, Transfer, Penalties)
        ↓
Results (Metrics, Plots, Tables)
```

---

## ⚙️ Features

* ✅ Deadline-aware scheduling using **sub-deadlines**
* ✅ Resource feasibility (CPU & RAM constraints)
* ✅ Marginal cost model:

  * Compute cost
  * Transfer cost
  * Contention penalty
* ✅ Billing-aware VM leasing (interval rounding)
* ✅ Slack propagation (USR)
* ✅ Multi-workflow benchmarking
* ✅ Statistical validation (Wilcoxon tests)


## 🔬 Core Algorithm (Simplified)

```text
1. Compute HEFT ranks
2. Initialize sub-deadlines
3. For each task:
   ├─ Select feasible VMs
   ├─ Compute EFT + marginal cost
   ├─ Enforce sub-deadline
   ├─ Pick lowest-cost VM
   └─ Assign task
4. Propagate slack (USR)
5. Compute final metrics
```

---

## 📊 Results

### 💰 Cost Reduction

* CETSS reduces cost vs:

  * HEFT
  * Min-Min
  * BPS-only
* Gains increase with **smaller billing intervals**

### ⚖️ Cost vs Makespan

* Slightly higher makespan
* Significantly lower cost
* Strong Pareto trade-off

### 🔍 Sensitivity

* Higher **α (deadline factor)** → better feasibility
* Stable across parameters

### 🧪 Ablation

* USR improves utilization and cost slightly
* BPS alone is not sufficient

---

## 🧪 Experiments

### Workflows

* CyberShake
* Epigenomics
* LIGO

### Parameters

| Parameter | Description                 |
| --------- | --------------------------- |
| α         | Deadline scaling factor     |
| γ         | Slack redistribution factor |
| μ         | Contention penalty weight   |
| Δ         | Billing interval            |

---

## ▶️ Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/cetss-scheduler.git
cd cetss-scheduler
```

---

### 2️⃣ Install dependencies

```bash
pip install numpy pandas scipy
```

---

### 3️⃣ Run experiments

```bash
python experiments.py
```


## 📈 Key Metrics

* 💰 Total Cost
* ⏱️ Makespan
* 📊 Deadline Satisfaction Rate
* ⚡ BPS Fraction
* 🖥️ VM Utilization
* 🔢 Number of VMs

---

## 🧩 Core Components

### 🔹 Scheduling Engine

* Cost-aware decision making
* Resource feasibility filtering
* Slack propagation

### 🔹 VM Model

* Time-aware CPU/RAM tracking
* Billing interval handling
* Load factor modeling

### 🔹 Experiment Suite

* Main comparison
* Sensitivity analysis
* Ablation study
* Statistical validation

---

## 📖 Contributions

* Novel **billing-aware scheduling framework**
* Integration of:

  * Cost optimization
  * Resource constraints
  * Slack redistribution
* Extensive benchmarking with statistical validation
