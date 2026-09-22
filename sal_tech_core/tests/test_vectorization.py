"""
SAL_tech Test 1: Deterministic State Verification
Validates CSR O(1) mathematical updates and proves Evacuation Agent never traverses dropped bridges.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from sal_tech_core.core.state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    STATUS_AVAILABLE,
    STATUS_BUSY
)
from sal_tech_core.agents.agent_evacuation import EvacuationAgent


def test_csr_o1_edge_update():
    """Verify that updating an edge updates the underlying CSR data in O(1)."""
    edges = [
        (0, 1, 5.0),
        (1, 2, 3.0),
        (2, 3, 2.0)
    ]
    matrix = StateMatrix(num_nodes=4, edges=edges)
    
    # Check that initial edge weight is 5.0
    idx_01 = matrix._edge_idx_map[(0, 1)]
    assert matrix.adj_matrix.data[idx_01] == 5.0

    # Update edge (0, 1) to 12.5 in O(1)
    matrix.update_edge_weight(0, 1, 12.5)
    assert matrix.adj_matrix.data[idx_01] == 12.5
    idx_10 = matrix._edge_idx_map[(1, 0)]
    assert matrix.adj_matrix.data[idx_10] == 12.5


def test_evacuation_never_crosses_dropped_bridge():
    """
    Feed the system a static matrix. Drop a specific bridge connecting Node A to Node B.
    Assertion: Prove that the Evacuation Agent never routes a civilian over the dropped bridge.
    If a route contains the dropped edge, the test FAILS.
    """
    # Graph topology:
    # 0 (Safe Zone) == [Bridge: 0-1 (weight 2.0)] == 1 == 2 (Civilian Cluster)
    # Alternative detour route: 0 == 3 (weight 4.0) == 4 (weight 4.0) == 2 (weight 1.0)
    edges = [
        (0, 1, 2.0),  # The critical bridge between Node 0 and Node 1
        (1, 2, 2.0),
        (0, 3, 4.0),
        (3, 4, 4.0),
        (4, 2, 1.0)
    ]
    matrix = StateMatrix(num_nodes=5, edges=edges)
    evac_agent = EvacuationAgent(matrix, safe_zone_nodes=[0])

    # Initial route for civilian cluster at Node 2 should cross Bridge (0, 1)
    routes_before = evac_agent.get_evacuation_routes(civilian_nodes=[2], safe_zone=0)
    route_before = routes_before[2]["route"].tolist()
    assert route_before == [2, 1, 0]  # Uses the bridge

    # DROP THE BRIDGE between Node 0 and Node 1
    bridge_node_a = 0
    bridge_node_b = 1
    matrix.update_edge_weight(bridge_node_a, bridge_node_b, np.inf)

    # Recalculate evacuation routes
    routes_after = evac_agent.get_evacuation_routes(civilian_nodes=[2], safe_zone=0)
    route_after = routes_after[2]["route"].tolist()
    distance_after = routes_after[2]["distance"]

    # Detour path must be: 2 -> 4 -> 3 -> 0 (total distance: 1.0 + 4.0 + 4.0 = 9.0)
    assert distance_after == 9.0
    assert route_after == [2, 4, 3, 0]

    # MATHEMATICAL PROOF: Check every consecutive edge in the route
    for i in range(len(route_after) - 1):
        edge = (route_after[i], route_after[i + 1])
        assert edge != (bridge_node_a, bridge_node_b), f"Route crossed dropped bridge {edge}!"
        assert edge != (bridge_node_b, bridge_node_a), f"Route crossed dropped bridge {edge}!"


def test_disconnected_island_isolation():
    """If all paths to safe zone are destroyed, node must be reported as ISOLATED."""
    edges = [
        (0, 1, 10.0)
    ]
    matrix = StateMatrix(num_nodes=2, edges=edges)
    evac_agent = EvacuationAgent(matrix, safe_zone_nodes=[0])

    # Drop the only connection
    matrix.update_edge_weight(0, 1, np.inf)
    routes = evac_agent.get_evacuation_routes([1], safe_zone=0)

    assert routes[1]["status"] == "ISOLATED"
    assert np.isinf(routes[1]["distance"])
    assert len(routes[1]["route"]) == 0
