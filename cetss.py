"""
CETSS scheduler: BPS + USR + Resource-Aware Marginal Cost.
Algorithm 1 + Equations from the paper (all bug-fixes applied).
"""
import math, copy
from typing import Dict, List
from core import (Task, WorkflowDAG, VMType, VMInstance,
                  exec_time, transfer_time, effective_bw,
                  compute_vm_cost, compute_upward_ranks,
                  load_factor_for_task)
from baselines import SchedulingState, _data_ready


def _resource_feasible(task, vm, at_time):
    """Eq. 15, time-aware."""
    return vm.cpu_free_at(at_time) >= task.cpu_req and \
           vm.ram_free_at(at_time) >= task.ram_req_mb


def _eft_cetss(task, vm, state, dag, task_map):
    """Eq. 13 with task-aware load factor."""
    t_data  = _data_ready(task, dag, task_map, vm, state)
    t_start = max(t_data, vm.start_time)
    lf = load_factor_for_task(vm, task.cpu_req, t_start)
    return t_start + exec_time(task, vm.vm_type) * lf


def _marginal_cost(task, vm, eft, dag, task_map, state, mu, now):
    """Eq. 17 – fixed to avoid double-counting contention and billing."""
    # Δ compute billing (Eq. 18)
    lease_e = vm.lease_end(now, state.billing_interval)
    delta_int = 0 if eft <= lease_e else math.ceil((eft - lease_e) / state.billing_interval)
    delta_compute = vm.vm_type.price_per_interval * delta_int

    # Transfer cost (Eq. 19)
    transfer = 0.0
    for pid in dag.predecessors(task.task_id):
        p = task_map[pid]
        data_mb = dag.data.get((pid, task.task_id), 0)
        if data_mb > 0 and p.assigned_vm is not None:
            pvm = next(v for v in state.vms if v.vm_id == p.assigned_vm)
            if pvm.location != vm.location:
                transfer += 0.01 * data_mb

    # Contention penalty (Eq. 20) – separate disincentive, not in billing
    t_data = _data_ready(task, dag, task_map, vm, state)
    t_start = max(t_data, vm.start_time)
    lf = load_factor_for_task(vm, task.cpu_req, t_start)
    contention = mu * max(0, lf - 1.0) * exec_time(task, vm.vm_type) * \
                 vm.vm_type.price_per_interval / state.billing_interval

    return delta_compute + transfer + contention


def _assign_cetss(task, vm, state, dag, task_map):
    """Finalize assignment with corrected load factor."""
    t_data  = _data_ready(task, dag, task_map, vm, state)
    t_start = max(t_data, vm.start_time)
    lf = load_factor_for_task(vm, task.cpu_req, t_start)
    t_fin   = t_start + exec_time(task, vm.vm_type) * lf
    task.assigned_vm = vm.vm_id
    task.start_time  = t_start
    task.finish_time = t_fin
    vm.task_history.append((task.task_id, t_start, t_fin, task.cpu_req, task.ram_req_mb))


def _init_subdeadlines(dag, task_map, vm_types, mbase, alpha):
    """Eqs. 8-11 with floor."""
    dglobal = alpha * mbase
    fastest = max(vm_types, key=lambda k: k.mips)
    max_bw  = max(k.bw_mbps for k in vm_types)
    avg_bw  = sum(k.bw_mbps for k in vm_types) / len(vm_types)
    avg_exec = {t.task_id: sum(exec_time(t, k) for k in vm_types) / len(vm_types)
                for t in dag.tasks}

    # Forward pass: EST
    est = {}
    for tid in dag.topo_order():
        preds = dag.predecessors(tid)
        if not preds: est[tid] = 0.0
        else:
            est[tid] = max(
                est[p] + exec_time(task_map[p], fastest) +
                transfer_time(dag.data.get((p,tid),0), max_bw)
                for p in preds)

    # Backward pass: subdeadlines
    sd = {}
    exits = {t.task_id for t in dag.tasks if not dag.successors(t.task_id)}
    for tid in reversed(dag.topo_order()):
        if tid in exits:
            sd[tid] = dglobal
        else:
            min_s = min(sd[s] - transfer_time(dag.data.get((tid,s),0), avg_bw)
                        for s in dag.successors(tid))
            sd[tid] = min_s - avg_exec[tid]
        sd[tid] = max(sd[tid], est[tid])   # floor (Eq. 11)

    for t in dag.tasks: t.subdeadline = sd[t.task_id]
    return dglobal, sd


def _propagate_slack(task, dag, task_map, dglobal, gamma, avg_bw):
    """Eq. 22 – divided equally across successors (multi-successor fix)."""
    slack = task.subdeadline - task.finish_time
    if slack <= 0: return
    succs = dag.successors(task.task_id)
    if not succs: return
    share = gamma * slack / len(succs)
    for sid in succs:
        c_ij = transfer_time(dag.data.get((task.task_id, sid), 0), avg_bw)
        s = task_map[sid]
        s.subdeadline = max(s.subdeadline, min(s.subdeadline + share, dglobal - c_ij))


def billed_utilization(vm, billing_interval):
    if not vm.task_history: return 0.0
    productive = sum(tf - ts for _, ts, tf, _, _ in vm.task_history)
    term = max(tf for _, _, tf, _, _ in vm.task_history)
    billed = math.ceil((term - vm.start_time) / billing_interval) * billing_interval
    return productive / max(billed, 1e-9)


def run_cetss(dag, vm_types, billing_interval,
              alpha=3.0, gamma=0.5, mu=1.0,
              enable_usr=True, resource_feasibility=True,
              seed=0, deadline_mode="soft", penalty_lambda=0.01):

    dag = copy.deepcopy(dag)
    task_map = {t.task_id: t for t in dag.tasks}
    state = SchedulingState(billing_interval)
    avg_bw = sum(k.bw_mbps for k in vm_types) / len(vm_types)
    ranks  = compute_upward_ranks(dag, vm_types)

    from baselines import run_heft as _heft
    mbase = _heft(copy.deepcopy(dag), vm_types, billing_interval, seed)["makespan"]
    dglobal, _ = _init_subdeadlines(dag, task_map, vm_types, mbase, alpha)

    bps_count = 0; infeasible_count = 0

    for tid in sorted(dag.topo_order(), key=lambda x: -ranks[x]):
        task = task_map[tid]
        t_data = max((task_map[p].finish_time for p in dag.predecessors(tid)), default=0.0)
        now    = t_data

        best_vm, best_cost, best_eft = None, float('inf'), float('inf')

        for vm in state.vms:
            t_start = max(t_data, vm.start_time)
            if resource_feasibility and not _resource_feasible(task, vm, t_start):
                infeasible_count += 1; continue
            eft  = _eft_cetss(task, vm, state, dag, task_map)
            if eft > task.subdeadline: continue
            cost = _marginal_cost(task, vm, eft, dag, task_map, state, mu, now)
            if cost < best_cost or (abs(cost-best_cost)<1e-9 and eft < best_eft):
                best_cost, best_eft, best_vm = cost, eft, vm

        if best_vm is None:
            best_type, best_cost_n, best_eft_n = None, float('inf'), float('inf')
            for vtype in vm_types:
                if resource_feasibility and (vtype.cpu_cores < task.cpu_req or
                                              vtype.ram_mb < task.ram_req_mb):
                    continue
                eft_n = max(0.0, t_data) + exec_time(task, vtype)
                if eft_n <= dglobal and vtype.price_per_interval < best_cost_n:
                    best_cost_n, best_eft_n, best_type = vtype.price_per_interval, eft_n, vtype
            if best_type is None:
                best_type = min(vm_types, key=lambda k: k.price_per_interval)
            best_vm = state.new_vm(best_type, max(0.0, t_data))

        # BPS tracking
        lease_e   = best_vm.lease_end(now, billing_interval)
        eft_final = _eft_cetss(task, best_vm, state, dag, task_map)
        if eft_final <= lease_e and best_vm.task_history:
            bps_count += 1

        _assign_cetss(task, best_vm, state, dag, task_map)

        if enable_usr:
            _propagate_slack(task, dag, task_map, dglobal, gamma, avg_bw)

    makespan = max(t.finish_time for t in dag.tasks)

    compute_cost = sum(
        compute_vm_cost(vm, max((tf for _,_,tf,_,_ in vm.task_history),
                                default=vm.start_time), billing_interval)
        for vm in state.vms)

    penalty = penalty_lambda * max(0, makespan - dglobal) if deadline_mode=="soft" else 0.0

    avg_util = (sum(billed_utilization(vm, billing_interval) for vm in state.vms)
                / max(1, len(state.vms)))

    return {
        "compute": compute_cost, "transfer": 0.0,
        "contention": 0.0, "penalty": penalty,
        "total": compute_cost + penalty,
        "makespan": makespan, "n_vms": len(state.vms),
        "deadline": dglobal, "met_deadline": makespan <= dglobal,
        "bps_fraction": bps_count / max(1, len(dag.tasks)),
        "infeasible_rejections": infeasible_count,
        "avg_billed_util": avg_util,
    }
