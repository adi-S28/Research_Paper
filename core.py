"""
Core simulation primitives - revised with time-aware resource tracking.
All equations reference the paper (Eq. N).
"""
import math, random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VMType:
    name: str
    mips: float
    cpu_cores: int
    ram_mb: int
    bw_mbps: float
    price_per_interval: float

VM_TYPES = [
    VMType("small",  1000, 1,  512,  100, 0.12),
    VMType("medium", 2000, 2,  1024, 200, 0.25),
    VMType("large",  4000, 4,  4096, 500, 0.48),
]

@dataclass
class Task:
    task_id: int
    mi: float
    cpu_req: int = 1
    ram_req_mb: int = 256
    assigned_vm: Optional[int] = None
    start_time: float = 0.0
    finish_time: float = 0.0
    subdeadline: float = float('inf')

@dataclass
class WorkflowDAG:
    tasks: List[Task]
    edges: Dict[int, List[int]]
    data: Dict[Tuple[int,int], float]
    name: str = "workflow"

    def predecessors(self, tid: int) -> List[int]:
        return [s for s, succs in self.edges.items() if tid in succs]
    def successors(self, tid: int) -> List[int]:
        return self.edges.get(tid, [])
    def topo_order(self) -> List[int]:
        in_deg = {t.task_id: 0 for t in self.tasks}
        for tid, succs in self.edges.items():
            for s in succs: in_deg[s] += 1
        q = [tid for tid, d in in_deg.items() if d == 0]
        order = []
        while q:
            node = q.pop(0); order.append(node)
            for s in self.edges.get(node, []):
                in_deg[s] -= 1
                if in_deg[s] == 0: q.append(s)
        return order

@dataclass
class VMInstance:
    vm_id: int
    vm_type: VMType
    location: str
    start_time: float
    # Timeline: list of (task_id, t_start, t_finish, cpu_req, ram_req)
    task_history: List[Tuple] = field(default_factory=list)

    def cpu_used_at(self, t: float) -> int:
        """CPU cores in use at time t (time-aware, Eq. 15)."""
        return sum(cr for _, ts, tf, cr, _ in self.task_history if ts <= t < tf)

    def ram_used_at(self, t: float) -> int:
        return sum(rr for _, ts, tf, _, rr in self.task_history if ts <= t < tf)

    def cpu_free_at(self, t: float) -> int:
        return self.vm_type.cpu_cores - self.cpu_used_at(t)

    def ram_free_at(self, t: float) -> int:
        return self.vm_type.ram_mb - self.ram_used_at(t)

    def utilization_at(self, t: float) -> float:
        return self.cpu_used_at(t) / self.vm_type.cpu_cores

    def avail_time(self) -> float:
        """Earliest time VM is fully idle."""
        if not self.task_history: return self.start_time
        return max(tf for _, _, tf, _, _ in self.task_history)

    def lease_end(self, now: float, billing_interval: float) -> float:
        """Eq. 16: paid-lease end at current time."""
        elapsed = now - self.start_time
        if elapsed <= 0: return self.start_time + billing_interval
        return self.start_time + math.ceil(elapsed / billing_interval) * billing_interval


def compute_vm_cost(vm: VMInstance, termination_time: float,
                    billing_interval: float) -> float:
    """Eq. 5."""
    duration = max(termination_time - vm.start_time, billing_interval)
    return vm.vm_type.price_per_interval * math.ceil(duration / billing_interval)

def effective_bw(src_loc, dst_loc, bw_cap, bw_used):
    if src_loc == dst_loc: return float('inf')
    return bw_cap * (1 - min(bw_used / bw_cap, 0.99))

def transfer_time(data_mb: float, eff_bw_mbps: float) -> float:
    if eff_bw_mbps == float('inf') or data_mb == 0: return 0.0
    return (data_mb * 8) / eff_bw_mbps

def load_factor(vm: VMInstance, at_time: float) -> float:
    """Eq. 12: load factor at a given scheduling time."""
    util = vm.utilization_at(at_time)
    if util >= 1.0: return 2.0
    return min(1.0 / (1.0 - util), 2.0)

def exec_time(task: Task, vm_type: VMType) -> float:
    return task.mi / vm_type.mips

def compute_upward_ranks(dag: WorkflowDAG, vm_types: List[VMType]) -> Dict[int, float]:
    avg_exec = {t.task_id: np.mean([exec_time(t, k) for k in vm_types])
                for t in dag.tasks}
    avg_bw = np.mean([k.bw_mbps for k in vm_types])
    rank: Dict[int, float] = {}
    for tid in reversed(dag.topo_order()):
        w = avg_exec[tid]
        succ_ranks = [transfer_time(dag.data.get((tid, s), 0), avg_bw) + rank.get(s, 0)
                      for s in dag.successors(tid)]
        rank[tid] = w + (max(succ_ranks) if succ_ranks else 0)
    return rank


def load_factor_for_task(vm: 'VMInstance', task_cpu: int, at_time: float) -> float:
    """
    Eq. 12 – revised: load factor = 1.0 when task fits in free cores (no oversubscription).
    Contention only when adding the task would exceed VM capacity.
    """
    used_after = vm.cpu_used_at(at_time) + task_cpu
    total = vm.vm_type.cpu_cores
    if used_after <= total:
        return 1.0          # fits cleanly – no contention
    util = used_after / total
    return min(1.0 / (1.0 - min(util * 0.5, 0.9)), 2.0)   # mild oversubscription model
