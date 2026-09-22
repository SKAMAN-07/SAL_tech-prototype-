"""
SAL_tech - Triage Agent
Linear programming resource allocation matching mass casualties to available medics via scipy.optimize.linprog.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Any, Optional
from ..core.state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    STATUS_AVAILABLE,
    STATUS_BUSY,
    STATUS_TIMEOUT,
    COL_ID,
    COL_STATUS
)
from .agent_evacuation import EvacuationAgent


class TriageAgent:
    """
    Linear Programming (LP) resource allocator:
    - Matches pending casualties to available medics.
    - Minimizes travel time while prioritizing high-severity casualties.
    - Locks dispatched medics to STATUS_BUSY.
    - Handles Fog of War timeouts by identifying offline medics and dispatching backups.
    """

    def __init__(
        self,
        state_matrix: StateMatrix,
        evac_agent: EvacuationAgent,
        severity_weight: float = 5.0,
        timeout_threshold_sec: float = 3600.0  # 60 simulated minutes
    ):
        self.state_matrix = state_matrix
        self.evac_agent = evac_agent
        self.severity_weight = severity_weight
        self.timeout_threshold_sec = timeout_threshold_sec

        # Active dispatch registry: casualty_id -> {medic_id, casualty_node, medic_node, assigned_time}
        self.active_dispatches: Dict[str, Dict[str, Any]] = {}
        # Reverse lookup: medic_id -> casualty_id
        self.medic_to_casualty: Dict[Any, str] = {}

    def allocate(
        self,
        casualties: Dict[str, Dict[str, Any]],
        sim_time: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Solves LP assignment problem to match AVAILABLE medics with PENDING casualties.
        Returns list of dispatch orders.
        """
        # Filter available medics
        avail_medics = self.state_matrix.get_available_assets(ASSET_TYPE_MEDIC)
        if len(avail_medics) == 0:
            return []

        # Filter pending casualties
        pending_keys = [k for k, v in casualties.items() if v.get("status") == "PENDING"]
        if not pending_keys:
            return []

        # Extract medic nodes
        medic_ids = []
        medic_nodes = []
        for row in avail_medics:
            idx = int(row[COL_ID])
            meta = self.state_matrix.asset_metadata.get(idx, {})
            medic_ids.append(meta.get("external_id", idx))
            medic_nodes.append(meta.get("node", 0))

        M = len(medic_ids)

        # Pre-filter top candidate casualties by severity (most critical first)
        # Cap candidate pool to min(len(pending), max(2 * M, 30)) for rapid sub-millisecond LP
        if len(pending_keys) > max(2 * M, 30):
            pending_keys.sort(key=lambda k: casualties[k].get("severity", 1.0), reverse=True)
            candidate_keys = pending_keys[:max(2 * M, 30)]
        else:
            candidate_keys = pending_keys

        casualty_nodes = [casualties[k]["node"] for k in candidate_keys]
        severities = np.array([casualties[k].get("severity", 1.0) for k in candidate_keys], dtype=np.float64)
        C = len(casualty_nodes)

        # Compute distance matrix between medics and casualties via Evac Agent
        dist_matrix = self.evac_agent.get_distance_matrix(medic_nodes, casualty_nodes)

        # Formulate LP cost matrix
        cost_matrix = dist_matrix.copy()
        
        # Severe penalty for unreachable nodes (np.inf in dist_matrix)
        unreachable_mask = np.isinf(cost_matrix)
        
        # Coverage reward ensures LP actively matches available medics rather than leaving them idle
        COVERAGE_REWARD = 10000.0
        cost_matrix -= (self.severity_weight * severities[np.newaxis, :] + COVERAGE_REWARD)
        cost_matrix[unreachable_mask] = 1e8  # Large penalty to forbid unreachable paths

        c_obj = cost_matrix.flatten()

        # Pure vectorized sparse constraint matrix using Kronecker product:
        # 1. Each medic assigned to at most 1 casualty: sum_j x[i, j] <= 1
        # 2. Each casualty receives at most 1 medic: sum_i x[i, j] <= 1
        A1 = sp.kron(sp.eye(M, format="csr"), np.ones((1, C)))
        A2 = sp.kron(np.ones((1, M)), sp.eye(C, format="csr"))
        A_ub = sp.vstack([A1, A2], format="csr")
        b_ub = np.ones(M + C, dtype=np.float64)

        bounds = [(0.0, 1.0) for _ in range(M * C)]

        # Run fast HiGHS simplex/interior-point linear program solver
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            return []

        x_sol = res.x.reshape((M, C))
        dispatch_orders = []

        for i in range(M):
            for j in range(C):
                if x_sol[i, j] > 0.5 and not unreachable_mask[i, j]:
                    m_id = medic_ids[i]
                    c_id = candidate_keys[j]
                    c_node = casualty_nodes[j]
                    m_node = medic_nodes[i]

                    # Lock medic to BUSY in State Matrix
                    self.state_matrix.update_asset_status(m_id, STATUS_BUSY)

                    casualties[c_id]["status"] = "ASSIGNED"
                    casualties[c_id]["assigned_medic"] = m_id

                    order = {
                        "medic_id": m_id,
                        "casualty_id": c_id,
                        "casualty_node": c_node,
                        "medic_node": m_node,
                        "assigned_time": sim_time,
                        "distance": float(dist_matrix[i, j]),
                        "severity": float(severities[j])
                    }
                    dispatch_orders.append(order)
                    self.active_dispatches[c_id] = order
                    self.medic_to_casualty[m_id] = c_id

        return dispatch_orders

    def check_timeouts(
        self,
        casualties: Dict[str, Dict[str, Any]],
        sim_time: float
    ) -> List[str]:
        """
        Detects Fog-of-War disconnected medics (heartbeat silent for > timeout_threshold_sec).
        Marks them TIMEOUT, frees the casualty, and returns list of casualty IDs needing backup.
        """
        orphaned_casualties: List[str] = []

        # Iterate over currently assigned medics
        for m_id, c_id in list(self.medic_to_casualty.items()):
            row_idx = self.state_matrix.asset_id_map.get(m_id)
            if row_idx is None:
                continue

            last_seen = self.state_matrix.asset_metadata.get(row_idx, {}).get("last_seen", 0.0)
            if (sim_time - last_seen) >= self.timeout_threshold_sec:
                # Medic timed out in Fog of War!
                self.state_matrix.update_asset_status(m_id, STATUS_TIMEOUT)
                
                # Free the casualty
                if c_id in casualties:
                    casualties[c_id]["status"] = "PENDING"
                    casualties[c_id]["assigned_medic"] = None
                    orphaned_casualties.append(c_id)

                if c_id in self.active_dispatches:
                    del self.active_dispatches[c_id]
                del self.medic_to_casualty[m_id]

        return orphaned_casualties
