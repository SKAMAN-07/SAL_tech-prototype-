"""
SAL_tech - Offline Crisis Orchestration Engine
Deterministic Main Execution Harness
"""

from __future__ import annotations
import sys
import os
import json
from typing import Tuple
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from sal_tech_core.core.state_matrix import (
        StateMatrix,
        ASSET_TYPE_MEDIC,
        ASSET_TYPE_TRUCK,
        STATUS_AVAILABLE
    )
    from sal_tech_core.core.orchestrator import Orchestrator
else:
    from .core.state_matrix import (
        StateMatrix,
        ASSET_TYPE_MEDIC,
        ASSET_TYPE_TRUCK,
        STATUS_AVAILABLE
    )
    from .core.orchestrator import Orchestrator


def create_demo_network() -> Tuple[StateMatrix, Orchestrator]:
    """Builds a small 10-node crisis grid with roads, medics, trucks, and camps."""
    # 10 Nodes: Node 0 is Safe Zone / Central Depot
    edges = [
        (0, 1, 4.0),
        (0, 2, 2.0),
        (1, 2, 1.0),
        (1, 3, 5.0),
        (2, 4, 3.0),
        (3, 4, 1.0),
        (3, 5, 6.0),
        (4, 6, 2.0),
        (5, 7, 3.0),
        (6, 7, 2.0),
        (6, 8, 4.0),
        (7, 9, 2.0),
        (8, 9, 3.0)
    ]
    matrix = StateMatrix(num_nodes=10, edges=edges)
    orchestrator = Orchestrator(matrix, safe_zones=[0])

    # Register assets
    matrix.register_asset("M1", ASSET_TYPE_MEDIC, 34.05, -118.24, status=STATUS_AVAILABLE, node=0)
    matrix.register_asset("M2", ASSET_TYPE_MEDIC, 34.06, -118.25, status=STATUS_AVAILABLE, node=2)
    matrix.register_asset("T1", ASSET_TYPE_TRUCK, 34.05, -118.24, status=STATUS_AVAILABLE, node=0, capacity=500.0, fuel=100.0)

    # Register civilian clusters and camps
    orchestrator.register_civilian_cluster(5)
    orchestrator.register_civilian_cluster(7)
    orchestrator.register_civilian_cluster(9)

    orchestrator.register_supply_camp("Camp_North", node=6, demand=200.0)
    orchestrator.register_supply_camp("Camp_East", node=8, demand=150.0)

    return matrix, orchestrator


def main():
    print("=" * 60)
    print("  SAL_tech - Offline Crisis Orchestration Engine")
    print("  Target: Entry-level CPUs / Mobile NPUs (Pure NumPy/SciPy)")
    print("=" * 60)

    matrix, orchestrator = create_demo_network()

    # Step 1: Initial Tick (Evacuation routes computed)
    print("\n[TICK 1] Initializing state & calculating initial evacuation routes...")
    plan = orchestrator.tick(dt=1.0)
    print(f"  Evacuation routes computed for {len(plan['evacuation_routes'])} civilian clusters.")
    for civ, rinfo in plan["evacuation_routes"].items():
        print(f"    Cluster {civ} -> Safe Zone 0: distance {rinfo['distance']:.1f}, route: {rinfo['route']}")

    # Step 2: Inject Road Destruction & Casualty Packets
    print("\n[TICK 2] Simulating earthquake events via LoRa telemetry...")
    orchestrator.queue.push({
        "type": "ROAD_UPDATE",
        "u": 6,
        "v": 7,
        "status": "DESTROYED"
    })
    orchestrator.queue.push({
        "type": "CASUALTY_REPORT",
        "id": "C_Severe_1",
        "node": 7,
        "severity": 5.0,
        "count": 2
    })
    orchestrator.queue.push({
        "type": "CASUALTY_REPORT",
        "id": "C_Moderate_2",
        "node": 5,
        "severity": 2.0,
        "count": 1
    })

    plan = orchestrator.tick(dt=5.0)
    print("  Road (6, 7) marked DESTROYED.")
    print("  Recalculated evacuation routes avoiding collapsed road (6, 7):")
    for civ, rinfo in plan["evacuation_routes"].items():
        print(f"    Cluster {civ} -> Safe Zone 0: distance {rinfo['distance']:.1f}, route: {rinfo['route']}")

    print(f"\n  Triage LP Allocations ({len(plan['triage_dispatches'])} dispatches):")
    for d in plan["triage_dispatches"]:
        print(f"    Medic {d['medic_id']} (at node {d['medic_node']}) -> Casualty {d['casualty_id']} (at node {d['casualty_node']}) | Dist: {d['distance']:.1f}, Severity: {d['severity']}")

    print(f"\n  Supply VRP Routes ({len(plan['supply_routes'])} routes):")
    for s in plan["supply_routes"]:
        print(f"    Truck {s['truck_id']} route: {s['route']} | Cargo: {s['cargo_delivered']} | Fuel Consumed: {s['fuel_consumed']:.1f}")

    orchestrator.periodic_state_dump()
    print("\n[SUCCESS] Deterministic state machine completed cycle. State dumped to disk.")


if __name__ == "__main__":
    main()
