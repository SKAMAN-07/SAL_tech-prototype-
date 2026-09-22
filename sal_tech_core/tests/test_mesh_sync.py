"""
SAL_tech Test 2: The Fog of War Mesh Disconnect
Simulates packet drops, 60-minute silence timeout, and automated backup medic reallocation.
"""

import pytest
import numpy as np
from sal_tech_core.core.state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    STATUS_AVAILABLE,
    STATUS_BUSY,
    STATUS_TIMEOUT
)
from sal_tech_core.core.orchestrator import Orchestrator


def test_fog_of_war_timeout_and_backup_reallocation():
    """
    Action: Deploy a Medic (M1) to a casualty zone. Silence their heartbeat for 60 simulated minutes.
    Assertion: Prove the Triage Agent registers a 'Timeout' and mathematically reallocates
    a backup Medic (M2) to the zone. State recovery must be 100% accurate.
    """
    edges = [
        (0, 1, 2.0),
        (1, 2, 2.0),
        (0, 3, 3.0),
        (3, 2, 3.0)
    ]
    matrix = StateMatrix(num_nodes=4, edges=edges)
    orchestrator = Orchestrator(matrix, safe_zones=[0])

    # Deploy Primary Medic M1 (at Node 1, closer to casualty at Node 2)
    matrix.register_asset("M1", ASSET_TYPE_MEDIC, 34.05, -118.24, status=STATUS_AVAILABLE, node=1, last_seen=0.0)
    # Deploy Backup Medic M2 (at Node 0, further from casualty at Node 2)
    matrix.register_asset("M2", ASSET_TYPE_MEDIC, 34.00, -118.20, status=STATUS_AVAILABLE, node=0, last_seen=0.0)

    # Inject casualty C1 at Node 2
    orchestrator.queue.push({
        "type": "CASUALTY_REPORT",
        "id": "C1",
        "node": 2,
        "severity": 4.0,
        "timestamp": 10.0
    })

    # Tick 1 at sim_time = 10.0: M1 should be selected (closer: dist 2.0 vs 4.0 for M2)
    plan1 = orchestrator.tick(dt=10.0)
    assert len(plan1["triage_dispatches"]) == 1
    d1 = plan1["triage_dispatches"][0]
    assert d1["medic_id"] == "M1"
    assert d1["casualty_id"] == "C1"
    assert matrix.asset_ledger[matrix.asset_id_map["M1"], 4] == STATUS_BUSY

    # SILENCE M1 FOR 60 SIMULATED MINUTES (3600 seconds)
    # Meanwhile, M2 sends occasional keepalive heartbeats
    orchestrator.queue.push({
        "type": "MEDIC_HEARTBEAT",
        "id": "M2",
        "loc": [34.00, -118.20],
        "status": "AVAILABLE",
        "node": 0,
        "timestamp": 3615.0
    })

    # Advance orchestrator time by 3605 seconds (sim_time reaches ~3615s > 60 min of silence for M1)
    plan2 = orchestrator.tick(dt=3605.0)

    # Verify M1 is flagged as STATUS_TIMEOUT in the State Matrix
    m1_row = matrix.asset_id_map["M1"]
    assert matrix.asset_ledger[m1_row, 4] == STATUS_TIMEOUT, "M1 was not marked TIMEOUT!"

    # Verify C1 was orphaned and reallocated to backup Medic M2
    assert "C1" in plan2["orphaned_casualties"]

    # Verify new dispatch order in plan
    dispatches = [d for d in plan2["triage_dispatches"] if d["casualty_id"] == "C1" and d["medic_id"] == "M2"]
    assert len(dispatches) >= 1, "Backup Medic M2 was not allocated to orphaned casualty C1!"

    # State recovery: M2 is now locked to BUSY, casualty is handled
    assert matrix.asset_ledger[matrix.asset_id_map["M2"], 4] == STATUS_BUSY
    assert orchestrator.parser.casualty_registry["C1"]["assigned_medic"] == "M2"
    assert orchestrator.parser.casualty_registry["C1"]["status"] == "ASSIGNED"
