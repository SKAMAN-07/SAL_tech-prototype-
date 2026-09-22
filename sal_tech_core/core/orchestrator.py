"""
SAL_tech - Orchestrator & Deterministic Event Loop
Coordinates telemetry queue, state matrix, and the mathematical agent swarm sequentially.
"""

from __future__ import annotations
import json
import numpy as np
from typing import Dict, List, Any, Optional
from .state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    ASSET_TYPE_TRUCK,
    STATUS_AVAILABLE,
    STATUS_BUSY,
    STATUS_TIMEOUT,
    NumpyEncoder
)
from .telemetry_parser import TelemetryQueue, TelemetryParser
from ..agents.agent_evacuation import EvacuationAgent
from ..agents.agent_triage import TriageAgent
from ..agents.agent_supply import SupplyAgent


class Orchestrator:
    """
    Deterministic State Machine event loop syncing agents and matrices sequentially.
    """

    def __init__(
        self,
        state_matrix: StateMatrix,
        safe_zones: Optional[List[int]] = None,
        action_plan_path: str = "action_plan.json",
        state_dump_path: str = "state_dump.json"
    ):
        self.state_matrix = state_matrix
        self.safe_zones = safe_zones if safe_zones is not None else [0]
        self.action_plan_path = action_plan_path
        self.state_dump_path = state_dump_path

        self.queue = TelemetryQueue()
        self.parser = TelemetryParser(self.state_matrix)

        # Initialize Agent Swarm
        self.evac_agent = EvacuationAgent(self.state_matrix, safe_zone_nodes=self.safe_zones)
        self.triage_agent = TriageAgent(self.state_matrix, self.evac_agent)
        self.supply_agent = SupplyAgent(self.state_matrix, self.evac_agent)

        self.sim_time: float = 0.0
        self.civilian_clusters: List[int] = []
        self.supply_camps: List[Dict[str, Any]] = []

        # Current Global Action Plan
        self.action_plan: Dict[str, Any] = {
            "sim_time": 0.0,
            "evacuation_routes": {},
            "triage_dispatches": [],
            "supply_routes": [],
            "orphaned_casualties": [],
            "status": "INITIALIZED"
        }

    def register_civilian_cluster(self, node: int) -> None:
        """Register a known civilian community node."""
        if node not in self.civilian_clusters:
            self.civilian_clusters.append(node)

    def register_supply_camp(self, camp_id: str, node: int, demand: float = 100.0) -> None:
        """Register a relief camp needing food/water/rations."""
        self.supply_camps.append({"id": camp_id, "node": node, "demand": demand})

    def tick(self, batch_size: int = 100, dt: float = 1.0) -> Dict[str, Any]:
        """
        Executes one deterministic state machine cycle:
        1. Read batch of telemetry packets from queue.
        2. Apply vectorized updates to StateMatrix.
        3. Trigger agents sequentially based on state diffs.
        4. Publish ActionPlan to disk.
        """
        self.sim_time += dt

        # 1. Read batch
        packets = self.queue.pop_batch(max_batch=batch_size)

        graph_changed = False
        casualty_changed = False
        supply_changed = False

        # 2. Update state matrix
        for p in packets:
            event_type, details = self.parser.parse_packet(p, sim_time=self.sim_time)
            if event_type == "GRAPH_CHANGED":
                graph_changed = True
            elif event_type == "CASUALTY_REPORTED":
                casualty_changed = True
            elif event_type == "TRUCK_UPDATED":
                supply_changed = True

        # Check Fog of War timeouts for medics
        orphaned = self.triage_agent.check_timeouts(self.parser.casualty_registry, sim_time=self.sim_time)
        if orphaned:
            casualty_changed = True

        # Initial pass if routes haven't been computed yet
        if not self.action_plan["evacuation_routes"] and self.civilian_clusters:
            graph_changed = True

        # 3. Trigger Agent Swarm
        # If road network changed, recalculate evacuation and re-evaluate routes
        if graph_changed:
            evac_routes = self.evac_agent.get_evacuation_routes(self.civilian_clusters, safe_zone=self.safe_zones[0])
            # Convert ndarrays to lists for JSON serialization
            self.action_plan["evacuation_routes"] = {
                str(k): {"distance": v["distance"], "route": v["route"].tolist(), "status": v["status"]}
                for k, v in evac_routes.items()
            }
            # Graph change might alter distances for triage and supply
            casualty_changed = True
            supply_changed = True

        if casualty_changed:
            new_dispatches = self.triage_agent.allocate(self.parser.casualty_registry, sim_time=self.sim_time)
            self.action_plan["triage_dispatches"].extend(new_dispatches)

        if supply_changed and self.supply_camps:
            new_supply_routes = self.supply_agent.plan_routes(self.supply_camps, depot_node=self.safe_zones[0])
            if new_supply_routes:
                self.action_plan["supply_routes"].extend(new_supply_routes)

        # 4. Publish Global Action Plan
        self.action_plan["sim_time"] = self.sim_time
        self.action_plan["orphaned_casualties"] = orphaned
        self.action_plan["status"] = "OK"

        self._publish_action_plan()
        return self.action_plan

    def _publish_action_plan(self) -> None:
        """Serializes current plan to local disk."""
        try:
            with open(self.action_plan_path, "w", encoding="utf-8") as f:
                json.dump(self.action_plan, f, indent=2, cls=NumpyEncoder)
        except Exception:
            pass

    def periodic_state_dump(self) -> None:
        """Simulate durable field checkpointing."""
        self.state_matrix.dump_state_to_file(self.state_dump_path)
