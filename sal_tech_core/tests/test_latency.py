"""
SAL_tech Test 3: Low-End Hardware Latency Benchmark
Validates that on a 10,000-node, 30,000-edge urban disaster graph with 500 casualties,
the Evacuation and Triage agents compute the global action plan in under 50ms.
"""

import pytest
import time
import numpy as np
import scipy.sparse as sp
from sal_tech_core.core.state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    STATUS_AVAILABLE
)
from sal_tech_core.agents.agent_evacuation import EvacuationAgent
from sal_tech_core.agents.agent_triage import TriageAgent


def generate_pseudo_city(num_nodes=10000, num_edges=30000, seed=42):
    """Generates a realistic urban sparse graph with 10k nodes and 30k edges."""
    rng = np.random.default_rng(seed)
    # Spanning backbone
    u_base = np.arange(num_nodes - 1, dtype=np.int32)
    v_base = u_base + 1
    w_base = rng.uniform(1.0, 10.0, size=num_nodes - 1)

    # Random urban grid cross-streets
    n_extra = num_edges - (num_nodes - 1)
    u_extra = rng.integers(0, num_nodes, size=n_extra, dtype=np.int32)
    v_extra = rng.integers(0, num_nodes, size=n_extra, dtype=np.int32)
    mask = u_extra != v_extra
    u_extra = u_extra[mask]
    v_extra = v_extra[mask]
    w_extra = rng.uniform(1.0, 15.0, size=len(u_extra))

    u_all = np.concatenate([u_base, u_extra])
    v_all = np.concatenate([v_base, v_extra])
    w_all = np.concatenate([w_base, w_extra])

    edges = list(zip(u_all, v_all, w_all))
    return StateMatrix(num_nodes=num_nodes, edges=edges)


@pytest.fixture(scope="module")
def urban_disaster_env():
    """Builds the 10k-node disaster environment once for benchmark runs."""
    matrix = generate_pseudo_city(num_nodes=10000, num_edges=30000, seed=123)
    evac_agent = EvacuationAgent(matrix, safe_zone_nodes=[0])
    triage_agent = TriageAgent(matrix, evac_agent, severity_weight=5.0)

    # Deploy 20 field medics across 4 major district dispatch sectors
    dispatch_hubs = [100, 1200, 3400, 8500]
    for i in range(20):
        matrix.register_asset(
            asset_id=f"M_{i}",
            asset_type=ASSET_TYPE_MEDIC,
            lat=34.0 + (i % 10) * 0.01,
            lon=-118.0 - (i // 10) * 0.01,
            status=STATUS_AVAILABLE,
            node=dispatch_hubs[i % len(dispatch_hubs)]
        )

    # Generate 500 simultaneous casualty reports
    rng = np.random.default_rng(999)
    cas_nodes = rng.choice(10000, size=500, replace=False)
    casualties = {}
    for j, c_node in enumerate(cas_nodes):
        casualties[f"C_{j}"] = {
            "id": f"C_{j}",
            "node": int(c_node),
            "severity": float(rng.uniform(1.0, 5.0)),
            "status": "PENDING"
        }

    return matrix, evac_agent, triage_agent, casualties, list(cas_nodes[:50])


def test_latency_sub_50ms(urban_disaster_env, benchmark):
    """
    Benchmark testing that Evacuation + Triage computation executes under 50 milliseconds.
    """
    matrix, evac_agent, triage_agent, casualties, civ_clusters = urban_disaster_env

    def run_cycle():
        # 1. Evacuation pathfinding for active civilian clusters
        routes = evac_agent.get_evacuation_routes(civ_clusters, safe_zone=0)

        # 2. Reset casualties to PENDING for benchmark iteration
        for c in casualties.values():
            c["status"] = "PENDING"
            c["assigned_medic"] = None
        # Reset medics to AVAILABLE
        for row in matrix.asset_ledger:
            row[4] = STATUS_AVAILABLE

        # 3. Triage LP allocation (20 medics x 500 casualties)
        dispatches = triage_agent.allocate(casualties, sim_time=100.0)
        return len(routes), len(dispatches)

    # Benchmark with pytest-benchmark
    benchmark.pedantic(run_cycle, iterations=1, rounds=10)

    # Direct timing measurement for assertion
    t_start = time.perf_counter()
    n_routes, n_dispatches = run_cycle()
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"\n[BENCHMARK] Elapsed Time: {elapsed_ms:.2f} ms | Routes: {n_routes} | Dispatches: {n_dispatches}")
    assert elapsed_ms < 50.0, f"Latency {elapsed_ms:.2f}ms exceeded strict 50ms threshold!"
