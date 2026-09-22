"""
SAL_tech - Evacuation Agent
Vectorized spatial optimization and dynamic Dijkstra pathfinding using scipy.sparse.csgraph.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse.csgraph as csg
from typing import Dict, List, Tuple, Any, Optional
from ..core.state_matrix import StateMatrix


class EvacuationAgent:
    """
    Computes rapid shortest paths from civilian clusters to safe zones.
    Dynamically responds to road destructions (weights set to infinity).
    """

    def __init__(self, state_matrix: StateMatrix, safe_zone_nodes: Optional[List[int]] = None):
        self.state_matrix = state_matrix
        self.safe_zone_nodes = safe_zone_nodes if safe_zone_nodes is not None else [0]
        self._last_dist_matrix: Optional[np.ndarray] = None
        self._last_pred_matrix: Optional[np.ndarray] = None

    def recalculate(self, source_nodes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes vectorized Dijkstra using scipy.sparse.csgraph.dijkstra.
        If source_nodes provided, computes only for those sources for maximum CPU speed.
        """
        indices = source_nodes if source_nodes is not None else self.safe_zone_nodes
        dist_matrix, pred_matrix = csg.dijkstra(
            csgraph=self.state_matrix.adj_matrix,
            directed=False,
            indices=indices,
            return_predecessors=True
        )
        self._last_dist_matrix = dist_matrix
        self._last_pred_matrix = pred_matrix
        return dist_matrix, pred_matrix

    def get_distance_matrix(self, from_nodes: List[int], to_nodes: List[int]) -> np.ndarray:
        """
        Computes an exact (len(from_nodes), len(to_nodes)) distance matrix.
        Optimized by running multi-source Dijkstra only for unique sources.
        """
        if not from_nodes or not to_nodes:
            return np.empty((len(from_nodes), len(to_nodes)), dtype=np.float64)

        from_arr = np.asarray(from_nodes, dtype=np.int32)
        to_arr = np.asarray(to_nodes, dtype=np.int32)
        unique_sources, inv = np.unique(from_arr, return_inverse=True)

        dists = csg.dijkstra(
            csgraph=self.state_matrix.adj_matrix,
            directed=False,
            indices=unique_sources,
            return_predecessors=False
        )
        # Re-index unique sources to original from_nodes order and slice target columns
        return dists[inv, :][:, to_arr]

    @staticmethod
    def reconstruct_path(predecessors: np.ndarray, source: int, target: int, source_idx: int = 0) -> np.ndarray:
        """
        Backtracks predecessor array to return route waypoints from source to target.
        Returns empty array if target is unreachable (predecessor is -9999).
        """
        if source == target:
            return np.array([source], dtype=np.int32)

        pred_row = predecessors[source_idx] if predecessors.ndim == 2 else predecessors
        path = []
        curr = target

        while curr != -9999:
            path.append(curr)
            if curr == source:
                break
            curr = int(pred_row[curr])

        if not path or path[-1] != source:
            return np.array([], dtype=np.int32)

        path.reverse()
        return np.array(path, dtype=np.int32)

    def get_evacuation_routes(self, civilian_nodes: List[int], safe_zone: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        Returns evacuation waypoints and travel costs for all civilian nodes to nearest safe zone.
        """
        target_safe_zone = safe_zone if safe_zone is not None else self.safe_zone_nodes[0]
        # Run Dijkstra from safe zone so one single-source pass covers all civilian nodes!
        dist_matrix, pred_matrix = csg.dijkstra(
            csgraph=self.state_matrix.adj_matrix,
            directed=False,
            indices=[target_safe_zone],
            return_predecessors=True
        )

        routes: Dict[int, Dict[str, Any]] = {}
        # Since graph is undirected, path from safe_zone to node reversed is path from node to safe_zone!
        for civ in civilian_nodes:
            d = dist_matrix[0, civ]
            if np.isinf(d):
                routes[civ] = {"distance": np.inf, "route": np.array([], dtype=np.int32), "status": "ISOLATED"}
            else:
                # Backtrack from civ to target_safe_zone
                path = EvacuationAgent.reconstruct_path(pred_matrix, source=target_safe_zone, target=civ, source_idx=0)
                # Reverse path to orient from civ to safe zone
                civ_to_safe_route = path[::-1] if path.size else np.array([], dtype=np.int32)
                routes[civ] = {
                    "distance": float(d),
                    "route": civ_to_safe_route,
                    "status": "EVACUATING"
                }

        return routes
