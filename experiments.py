"""Full experiment suite with Wilcoxon tests (Table II parameters)."""
import sys, os
sys.path.insert(0, '/home/claude/sim')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import product
import warnings; warnings.filterwarnings("ignore")
from core import VM_TYPES
from workflows import make_workflow
from baselines import run_heft, run_minmin
from cetss import run_cetss

SEEDS = list(range(10))
WORKFLOWS = [("CyberShake",50), ("Epigenomics",50), ("LIGO",49)]
BILLING_INTERVALS = [3600, 300, 60]
ALPHA_VALUES  = [1.5, 2.0, 3.0, 5.0]
GAMMA_VALUES  = [0.25, 0.5, 1.0]
MU_VALUES     = [0.0, 0.5, 1.0, 2.0]
DEFAULT_ALPHA = 3.0
DEFAULT_GAMMA = 0.5
DEFAULT_MU    = 1.0

def safe_wilcoxon(x, y):
    diff = np.array(x) - np.array(y)
    if np.all(diff == 0): return 1.0, 0.0
    try:
        stat, p = wilcoxon(diff, alternative='less')
        n = len(diff); r = 1 - (2*stat)/(n*(n+1))
        return float(p), float(r)
    except: return 1.0, 0.0


def run_main_comparison():
    records = []
    total = len(WORKFLOWS)*len(BILLING_INTERVALS)*len(SEEDS)
    done  = 0
    for (wf,n), delta in product(WORKFLOWS, BILLING_INTERVALS):
        for seed in SEEDS:
            dag = make_workflow(wf, n, seed)
            rh  = run_heft(dag, VM_TYPES, delta, seed)
            rm  = run_minmin(dag, VM_TYPES, delta, seed)
            rb  = run_cetss(dag, VM_TYPES, delta, alpha=DEFAULT_ALPHA,
                             gamma=DEFAULT_GAMMA, mu=DEFAULT_MU,
                             enable_usr=False, seed=seed)
            rc  = run_cetss(dag, VM_TYPES, delta, alpha=DEFAULT_ALPHA,
                             gamma=DEFAULT_GAMMA, mu=DEFAULT_MU,
                             enable_usr=True, seed=seed)
            records.append({
                "workflow":wf,"n_tasks":n,"billing_interval":delta,"seed":seed,
                "heft_cost":rh["total"],"minmin_cost":rm["total"],
                "bps_cost":rb["total"],"cetss_cost":rc["total"],
                "heft_ms":rh["makespan"],"minmin_ms":rm["makespan"],
                "cetss_ms":rc["makespan"],"heft_nvms":rh["n_vms"],
                "cetss_nvms":rc["n_vms"],"cetss_bps_frac":rc["bps_fraction"],
                "cetss_util":rc["avg_billed_util"],"cetss_met_dl":int(rc["met_deadline"]),
                "cetss_infeasible":rc["infeasible_rejections"],
            })
            done+=1
            if done%30==0: print(f"  main: {done}/{total}", flush=True)
    return pd.DataFrame(records)


def run_ablation():
    records = []
    for (wf,n), gamma, seed in product(WORKFLOWS, GAMMA_VALUES, SEEDS):
        dag = make_workflow(wf, n, seed)
        rb  = run_cetss(dag, VM_TYPES, 300, alpha=DEFAULT_ALPHA, gamma=0.0,
                         mu=DEFAULT_MU, enable_usr=False, seed=seed)
        rc  = run_cetss(dag, VM_TYPES, 300, alpha=DEFAULT_ALPHA, gamma=gamma,
                         mu=DEFAULT_MU, enable_usr=True, seed=seed)
        records.append({"workflow":wf,"gamma":gamma,"seed":seed,
                         "bps_cost":rb["total"],"bps_ms":rb["makespan"],
                         "full_cost":rc["total"],"full_ms":rc["makespan"],
                         "bps_frac":rc["bps_fraction"],"util":rc["avg_billed_util"]})
    return pd.DataFrame(records)


def run_sensitivity():
    records = []
    wf, n = "CyberShake", 50
    for delta, alpha, seed in product(BILLING_INTERVALS, ALPHA_VALUES, SEEDS):
        dag = make_workflow(wf, n, seed)
        r   = run_cetss(dag, VM_TYPES, delta, alpha=alpha,
                         gamma=DEFAULT_GAMMA, mu=DEFAULT_MU, seed=seed)
        records.append({"param":"alpha_delta","delta":delta,"alpha":alpha,
                         "mu":DEFAULT_MU,"gamma":DEFAULT_GAMMA,"seed":seed,
                         **{k:r[k] for k in ("total","compute","makespan","n_vms",
                                              "bps_fraction","avg_billed_util","met_deadline")}})
    for mu, seed in product(MU_VALUES, SEEDS):
        dag = make_workflow(wf, n, seed)
        r   = run_cetss(dag, VM_TYPES, 300, alpha=DEFAULT_ALPHA,
                         gamma=DEFAULT_GAMMA, mu=mu, seed=seed)
        records.append({"param":"mu","delta":300,"alpha":DEFAULT_ALPHA,
                         "mu":mu,"gamma":DEFAULT_GAMMA,"seed":seed,
                         **{k:r[k] for k in ("total","compute","makespan","n_vms",
                                              "bps_fraction","avg_billed_util","met_deadline")}})
    return pd.DataFrame(records)


def compute_stats(df):
    summary, ms_rows = [], []
    for (wf, delta), g in df.groupby(["workflow","billing_interval"]):
        for algo, col in [("HEFT","heft_cost"),("Min-Min","minmin_cost"),
                           ("BPS-only","bps_cost"),("CETSS","cetss_cost")]:
            v = g[col].values
            summary.append({"workflow":wf,"billing_interval":delta,"algorithm":algo,
                             "mean":np.mean(v),"std":np.std(v),"median":np.median(v)})
        p_h,r_h = safe_wilcoxon(g.cetss_cost.values, g.heft_cost.values)
        p_m,r_m = safe_wilcoxon(g.cetss_cost.values, g.minmin_cost.values)
        p_b,r_b = safe_wilcoxon(g.cetss_cost.values, g.bps_cost.values)
        summary.append({"workflow":wf,"billing_interval":delta,"algorithm":"STATS",
                         "p_heft":p_h,"r_heft":r_h,"p_minmin":p_m,"r_minmin":r_m,
                         "p_bps":p_b,"r_bps":r_b})
        ms_rows.append({"workflow":wf,"billing_interval":delta,
                         "heft_ms":g.heft_ms.mean(),"cetss_ms":g.cetss_ms.mean(),
                         "dl_rate":g.cetss_met_dl.mean(),"bps_frac":g.cetss_bps_frac.mean(),
                         "util":g.cetss_util.mean(),"cetss_nvms":g.cetss_nvms.mean(),
                         "heft_nvms":g.heft_nvms.mean()})
    return pd.DataFrame(summary), pd.DataFrame(ms_rows)


if __name__ == "__main__":
    os.makedirs("/home/claude/sim/results", exist_ok=True)
    print("=== Main Comparison ===")
    df_main = run_main_comparison(); df_main.to_csv("/home/claude/sim/results/main.csv",index=False)
    print(f"  {len(df_main)} rows")
    print("=== Ablation ===")
    df_abl = run_ablation(); df_abl.to_csv("/home/claude/sim/results/ablation.csv",index=False)
    print(f"  {len(df_abl)} rows")
    print("=== Sensitivity ===")
    df_sens = run_sensitivity(); df_sens.to_csv("/home/claude/sim/results/sensitivity.csv",index=False)
    print(f"  {len(df_sens)} rows")
    print("=== Stats ===")
    df_sum, df_ms = compute_stats(df_main)
    df_sum.to_csv("/home/claude/sim/results/summary.csv",index=False)
    df_ms.to_csv("/home/claude/sim/results/makespan.csv",index=False)
    print("Done.")
