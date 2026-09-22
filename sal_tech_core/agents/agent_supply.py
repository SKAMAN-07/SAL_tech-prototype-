"""
SAL_tech - Supply Agent
Constrained Vehicle Routing Problem (VRP) solver using vectorized Simulated Annealing (2-opt).
Routes trucks with finite capacities to multiple disaster camps, avoiding destroyed edges.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..core.state_matrix import (
    StateMatrix,
    ASSET_TYPE_TRUCK,
    STATUS_AVAILABLE,
    STATUS_BUSY,
    COL_ID,
    COL_STATUS,
    COL_CAPACITY
)
from .agent_evacuation import EvacuationAgent


class SupplyAgent:
    """
    Solves Capacitated VRP under fuel and connectivity constraints using Simulated Annealing.
    """

    def __init__(
        self,
        state_matrix: StateMatrix,
        evac_agent: EvacuationAgent,
        fuel_consumption_rate: float = 0.2  # Fuel burned per distance unit
    ):
        self.state_matrix = state_matrix
        self.evac_agent = evac_agent
        self.fuel_consumption_rate = fuel_consumption_rate
        self.active_deliveries: Dict[str, Dict[str, Any]] = {}

    def plan_routes(
        self,
        camps: List[Dict[str, Any]],  # List of {"id": str, "node": int, "demand": float}
        depot_node: int = 0,
        max_iterations: int = 500,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Plans VRP routes for available trucks delivering rations to disaster camps.
        Returns list of truck routes with waypoints, fuel consumption, and load.
        """
        avail_trucks = self.state_matrix.get_available_assets(ASSET_TYPE_TRUCK)
        if len(avail_trucks) == 0 or not camps:
            return []

        routes_output: List[Dict[str, Any]] = []
        pending_camps = list(camps)

        # Iterate over available trucks
        for t_row in avail_trucks:
            if not pending_camps:
                break

            idx = int(t_row[COL_ID])
            meta = self.state_matrix.asset_metadata.get(idx, {})
            t_id = meta.get("external_id", idx)
            start_node = meta.get("node", depot_node)
            capacity = float(t_row[COL_CAPACITY])
            available_fuel = float(meta.get("fuel", 100.0))

            # Greedily assign camps up to truck capacity
            current_load = 0.0
            assigned_camps = []
            remaining_camps = []

            for camp in pending_camps:
                d = float(camp.get("demand", 100.0))
                if (current_load + d) <= capacity:
                    current_load += d
                    assigned_camps.append(camp)
                else:
                    remaining_camps.append(camp)

            if not assigned_camps:
                continue

            # Optimize visiting order using Simulated Annealing
            camp_nodes = [c["node"] for c in assigned_camps]
            all_stops = [start_node] + camp_nodes + [start_node]

            # Get pairwise distance matrix
            unique_nodes = list(dict.fromkeys(all_stops))
            node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
            dist_submatrix = self.evac_agent.get_distance_matrix(unique_nodes, unique_nodes)

            # Check if any camp is unreachable from depot
            tour_indices = [node_to_idx[n] for n in all_stops]
            best_tour = self._simulated_annealing_tour(
                tour_indices,
                dist_submatrix,
                max_iterations=max_iterations,
                t0=initial_temp,
                cooling=cooling_rate
            )

            # Reconstruct sequence of graph nodes
            route_nodes = [unique_nodes[i] for i in best_tour]
            route_distance = self._calculate_tour_distance(best_tour, dist_submatrix)

            if np.isinf(route_distance):
                # Unreachable route due to destroyed bridges; cannot dispatch this plan
                continue

            fuel_needed = route_distance * self.fuel_consumption_rate
            if fuel_needed > available_fuel:
                # Truck does not have enough fuel for round trip
                continue

            # Valid route found!
            pending_camps = remaining_camps
            new_fuel = max(0.0, available_fuel - fuel_needed)
            meta["fuel"] = new_fuel

            # Lock truck status to BUSY
            self.state_matrix.update_asset_status(t_id, STATUS_BUSY)

            delivery_record = {
                "truck_id": t_id,
                "route": route_nodes,
                "total_distance": float(route_distance),
                "fuel_consumed": float(fuel_needed),
                "remaining_fuel": float(new_fuel),
                "cargo_delivered": float(current_load),
                "camps_visited": [c["id"] for c in assigned_camps]
            }
            routes_output.append(delivery_record)
            self.active_deliveries[t_id] = delivery_record

        return routes_output

    def _calculate_tour_distance(self, tour: List[int], dist_matrix: np.ndarray) -> float:
        """Vectorized summation of tour distances."""
        u_arr = tour[:-1]
        v_arr = tour[1:]
        return float(np.sum(dist_matrix[u_arr, v_arr]))

    def _simulated_annealing_tour(
        self,
        initial_tour: List[int],
        dist_matrix: np.ndarray,
        max_iterations: int,
        t0: float,
        cooling: float
    ) -> List[int]:
        """
        Simulated Annealing with 2-opt perturbations to optimize TSP tour.
        Fixed depot at start and end.
        """
        n_stops = len(initial_tour)
        if n_stops <= 3:
            return list(initial_tour)

        current_tour = list(initial_tour)
        current_cost = self._calculate_tour_distance(current_tour, dist_matrix)
        best_tour = list(current_tour)
        best_cost = current_cost

        T = t0
        for _ in range(max_iterations):
            # Pick two random intermediate indices for 2-opt reversal
            i, j = np.random.randint(1, n_stops - 1, size=2)
            if i > j:
                i, j = j, i
            elif i == j:
                continue

            # Candidate tour with segment [i:j+1] reversed
            candidate_tour = current_tour[:i] + current_tour[i : j + 1][::-1] + current_tour[j + 1 :]
            candidate_cost = self._calculate_tour_distance(candidate_tour, dist_matrix)

            delta = candidate_cost - current_cost

            # Acceptance criterion
            if delta < 0 or (delta < 1e5 and np.random.rand() < np.exp(-delta / max(T, 1e-4))):
                current_tour = candidate_tour
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best_tour = list(current_tour)
                    best_cost = current_cost

            T *= cooling
            if T < 1e-3:
                break

        return best_tour
