"""
SAL_tech Test 4: End-to-End Mass Casualty Simulation (The GCSP Validation)
Simulates a 10-minute earthquake scenario with 1,000 randomized LoRa telemetry packets.
Asserts:
1. All casualties are assigned a medic.
2. No trucks have negative fuel.
3. No civilian evacuation routes pass through destroyed infrastructure.
"""

import pytest
import numpy as np
from sal_tech_core.core.state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    ASSET_TYPE_TRUCK,
    STATUS_AVAILABLE,
    STATUS_BUSY,
    COL_STATUS
)
from sal_tech_core.core.orchestrator import Orchestrator


def build_disaster_grid(num_nodes=60, num_edges=220, seed=42):
    """Builds a connected urban crisis grid with arterial roads and bypasses."""
    rng = np.random.default_rng(seed)
    u_base = np.arange(num_nodes - 1, dtype=np.int32)
    v_base = u_base + 1
    w_base = rng.uniform(2.0, 8.0, size=num_nodes - 1)

    n_extra = num_edges - (num_nodes - 1)
    u_extra = rng.integers(0, num_nodes, size=n_extra, dtype=np.int32)
    v_extra = rng.integers(0, num_nodes, size=n_extra, dtype=np.int32)
    mask = u_extra != v_extra
    u_extra = u_extra[mask]
    v_extra = v_extra[mask]
    w_extra = rng.uniform(2.0, 12.0, size=len(u_extra))

    edges = list(zip(
        np.concatenate([u_base, u_extra]),
        np.concatenate([v_base, v_extra]),
        np.concatenate([w_base, w_extra])
    ))
    matrix = StateMatrix(num_nodes=num_nodes, edges=edges)
    return matrix, edges


def test_e2e_disaster_simulation():
    """
    Action: Simulate a 10-minute earthquake scenario. Inject 1,000 randomized telemetry packets.
    Assertion:
    1. All casualties are assigned a medic.
    2. No trucks have negative fuel.
    3. No civilian routes pass through destroyed infrastructure.
    """
    num_nodes = 60
    matrix, edges = build_disaster_grid(num_nodes=num_nodes, num_edges=220, seed=777)
    orchestrator = Orchestrator(matrix, safe_zones=[0])

    # Register civilian community hubs
    civilian_nodes = [5, 12, 18, 25, 34, 42, 50, 58]
    for c_node in civilian_nodes:
        orchestrator.register_civilian_cluster(c_node)

    # Register relief camps
    orchestrator.register_supply_camp("Camp_Alpha", node=15, demand=120.0)
    orchestrator.register_supply_camp("Camp_Beta", node=30, demand=150.0)
    orchestrator.register_supply_camp("Camp_Gamma", node=45, demand=100.0)

    # Deploy 40 Medics across hubs
    for i in range(40):
        matrix.register_asset(
            asset_id=f"MEDIC_{i}",
            asset_type=ASSET_TYPE_MEDIC,
            lat=34.05 + (i % 10) * 0.01,
            lon=-118.25 - (i // 10) * 0.01,
            status=STATUS_AVAILABLE,
            node=int((i * 3) % num_nodes),
            last_seen=0.0
        )

    # Deploy 6 Supply Trucks with full fuel
    for j in range(6):
        matrix.register_asset(
            asset_id=f"TRUCK_{j}",
            asset_type=ASSET_TYPE_TRUCK,
            lat=34.05,
            lon=-118.25,
            status=STATUS_AVAILABLE,
            node=0,
            capacity=500.0,
            fuel=250.0,
            last_seen=0.0
        )

    rng = np.random.default_rng(2026)

    # Pre-generate 1,000 randomized LoRa packets
    packets = []
    # 1. 200 Road update packets
    candidate_edges = list(matrix._edge_idx_map.keys())
    for _ in range(200):
        u, v = candidate_edges[rng.integers(0, len(candidate_edges))]
        # 40% chance destroyed, 60% clear/repaired
        status = "DESTROYED" if rng.random() < 0.4 else "CLEAR"
        packets.append({
            "type": "ROAD_UPDATE",
            "u": int(u),
            "v": int(v),
            "status": status,
            "weight": 5.0
        })

    # 2. 35 casualty reports (total casualties manageable by our 40-medic fleet)
    for c_idx in range(35):
        c_node = int(rng.integers(1, num_nodes))
        packets.append({
            "type": "CASUALTY_REPORT",
            "id": f"CASUALTY_{c_idx}",
            "node": c_node,
            "severity": float(rng.uniform(1.0, 5.0)),
            "count": 1
        })

    # 3. 565 Medic heartbeats & 200 Truck heartbeats (simulating continuous mesh telemetry)
    for k in range(565):
        m_idx = k % 40
        packets.append({
            "type": "MEDIC_HEARTBEAT",
            "id": f"MEDIC_{m_idx}",
            "loc": [34.05, -118.25],
            "node": int((m_idx * 3) % num_nodes)
        })

    for t_idx in range(200):
        tr_id = f"TRUCK_{t_idx % 6}"
        packets.append({
            "type": "TRUCK_HEARTBEAT",
            "id": tr_id,
            "loc": [34.05, -118.25],
            "node": 0,
            "fuel": float(rng.uniform(80.0, 250.0)),
            "capacity": 500.0
        })

    # Shuffle packets to simulate realistic async decentralized radio mesh arrival
    rng.shuffle(packets)
    assert len(packets) == 1000

    # Distribute 1,000 packets across a 10-minute simulation (60 ticks, dt=10.0s)
    num_ticks = 60
    batch_size = len(packets) // num_ticks  # ~16 packets per tick

    pkt_pointer = 0
    final_plan = None
    for tick_num in range(num_ticks):
        # Push batch to LoRa queue
        next_batch = packets[pkt_pointer : pkt_pointer + batch_size]
        pkt_pointer += batch_size
        for p in next_batch:
            p["timestamp"] = tick_num * 10.0
            orchestrator.queue.push(p)

        # Run deterministic tick
        final_plan = orchestrator.tick(batch_size=len(next_batch), dt=10.0)

    # Push any leftover packets
    while pkt_pointer < len(packets):
        orchestrator.queue.push(packets[pkt_pointer])
        pkt_pointer += 1
    final_plan = orchestrator.tick(batch_size=100, dt=10.0)

    # =========================================================================
    # GCSP RIGOROUS ASSERTIONS
    # =========================================================================

    # Assertion 1: All casualties must be assigned a medic
    casualties = orchestrator.parser.casualty_registry
    assert len(casualties) == 35, f"Expected 35 casualties, found {len(casualties)}"
    unassigned = [c_id for c_id, c in casualties.items() if c.get("status") != "ASSIGNED"]
    assert len(unassigned) == 0, f"Found unassigned casualties at simulation end: {unassigned}"

    # Assertion 2: No trucks have negative fuel
    for row in matrix.asset_ledger:
        if row[1] == ASSET_TYPE_TRUCK:
            idx = int(row[0])
            meta = matrix.asset_metadata.get(idx, {})
            fuel = meta.get("fuel", 0.0)
            assert fuel >= 0.0, f"Truck {meta.get('external_id')} has negative fuel: {fuel}"

    for route_info in final_plan.get("supply_routes", []):
        assert route_info.get("remaining_fuel", 0.0) >= 0.0, "Supply route resulted in negative remaining fuel!"

    # Assertion 3: No civilian evacuation routes pass through destroyed infrastructure
    evac_routes = final_plan.get("evacuation_routes", {})
    for civ_id, rinfo in evac_routes.items():
        route = rinfo.get("route", [])
        if len(route) >= 2:
            for k in range(len(route) - 1):
                u, v = route[k], route[k + 1]
                idx_uv = matrix._edge_idx_map.get((u, v))
                if idx_uv is not None:
                    w = matrix.adj_matrix.data[idx_uv]
                    assert not np.isinf(w), (
                        f"Evacuation route for Cluster {civ_id} passes through DESTROYED road ({u}, {v})!"
                    )

    # Verify atomic state dump
    orchestrator.periodic_state_dump()
