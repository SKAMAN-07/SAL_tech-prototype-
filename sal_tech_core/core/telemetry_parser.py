"""
SAL_tech - Telemetry Ingestion Engine
Simulates decentralized LoRa radio mesh network packets and parses them into StateMatrix updates.
"""

from __future__ import annotations
import collections
import json
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from .state_matrix import (
    StateMatrix,
    ASSET_TYPE_MEDIC,
    ASSET_TYPE_TRUCK,
    STATUS_AVAILABLE,
    STATUS_BUSY,
    STATUS_TIMEOUT
)


class TelemetryQueue:
    """In-memory thread-safe FIFO queue simulating LoRa mesh buffer."""

    def __init__(self):
        self._queue = collections.deque()

    def push(self, packet: Union[Dict[str, Any], str, bytes]) -> None:
        """Enqueue incoming radio packet."""
        if isinstance(packet, (str, bytes)):
            packet = json.loads(packet)
        self._queue.append(packet)

    def pop_batch(self, max_batch: int = 100) -> List[Dict[str, Any]]:
        """Drain up to max_batch packets sequentially."""
        batch = []
        while self._queue and len(batch) < max_batch:
            batch.append(self._queue.popleft())
        return batch

    def __len__(self) -> int:
        return len(self._queue)


class TelemetryParser:
    """
    Parses LoRa mesh telemetry packets and applies vectorized updates to StateMatrix.
    """

    def __init__(self, state_matrix: StateMatrix):
        self.state_matrix = state_matrix
        self.road_id_to_edge: Dict[int, Tuple[int, int]] = {}
        self.casualty_registry: Dict[str, Dict[str, Any]] = {}

    def register_road_id(self, road_id: int, u: int, v: int) -> None:
        """Maps LoRa infrastructure ID to graph edge (u, v)."""
        self.road_id_to_edge[road_id] = (u, v)

    def parse_packet(self, packet: Dict[str, Any], sim_time: float = 0.0) -> Tuple[str, Any]:
        """
        Translates packet into state update and returns event tuple (event_type, details).
        Supported types:
        - ROAD_UPDATE
        - MEDIC_HEARTBEAT
        - CASUALTY_REPORT
        - TRUCK_HEARTBEAT
        """
        p_type = packet.get("type", "").upper()

        if p_type == "ROAD_UPDATE":
            u, v = None, None
            if "id" in packet and packet["id"] in self.road_id_to_edge:
                u, v = self.road_id_to_edge[packet["id"]]
            elif "u" in packet and "v" in packet:
                u, v = int(packet["u"]), int(packet["v"])

            if u is not None and v is not None:
                status = str(packet.get("status", "")).upper()
                if status == "DESTROYED":
                    weight = np.inf
                elif status in ("CLEAR", "REPAIRED"):
                    weight = float(packet.get("weight", 1.0))
                else:
                    weight = float(packet.get("weight", np.inf))

                self.state_matrix.update_edge_weight(u, v, weight)
                return ("GRAPH_CHANGED", {"u": u, "v": v, "weight": weight, "status": status})

        elif p_type == "MEDIC_HEARTBEAT":
            medic_id = packet["id"]
            loc = packet.get("loc", [0.0, 0.0])
            node = packet.get("node")
            t = packet.get("timestamp", sim_time)

            if medic_id in self.state_matrix.asset_id_map:
                row_idx = self.state_matrix.asset_id_map[medic_id]
                curr_status = int(self.state_matrix.asset_ledger[row_idx, 4])
                if "status" in packet:
                    status_str = str(packet["status"]).upper()
                    status = STATUS_BUSY if status_str == "BUSY" else STATUS_AVAILABLE
                else:
                    status = curr_status

                self.state_matrix.update_asset_location(medic_id, loc[0], loc[1], node=node, timestamp=t)
                self.state_matrix.update_asset_status(medic_id, status)
            else:
                status_str = str(packet.get("status", "AVAILABLE")).upper()
                status = STATUS_BUSY if status_str == "BUSY" else STATUS_AVAILABLE
                self.state_matrix.register_asset(
                    asset_id=medic_id,
                    asset_type=ASSET_TYPE_MEDIC,
                    lat=loc[0],
                    lon=loc[1],
                    status=status,
                    capacity=1.0,
                    node=node,
                    last_seen=t
                )
            return ("MEDIC_UPDATED", {"id": medic_id, "status": status, "timestamp": t})

        elif p_type == "CASUALTY_REPORT":
            c_id = packet.get("id", f"C_{len(self.casualty_registry)}")
            node = int(packet["node"])
            severity = float(packet.get("severity", 1.0))
            count = int(packet.get("count", 1))
            t = packet.get("timestamp", sim_time)

            self.casualty_registry[c_id] = {
                "id": c_id,
                "node": node,
                "severity": severity,
                "count": count,
                "status": "PENDING",
                "assigned_medic": None,
                "timestamp": t
            }
            return ("CASUALTY_REPORTED", self.casualty_registry[c_id])

        elif p_type == "TRUCK_HEARTBEAT":
            truck_id = packet["id"]
            loc = packet.get("loc", [0.0, 0.0])
            node = packet.get("node")
            fuel = float(packet.get("fuel", 100.0))
            capacity = float(packet.get("capacity", 500.0))
            t = packet.get("timestamp", sim_time)

            if truck_id in self.state_matrix.asset_id_map:
                row_idx = self.state_matrix.asset_id_map[truck_id]
                curr_status = int(self.state_matrix.asset_ledger[row_idx, 4])
                if "status" in packet:
                    status_str = str(packet["status"]).upper()
                    status = STATUS_BUSY if status_str == "BUSY" else STATUS_AVAILABLE
                else:
                    status = curr_status

                self.state_matrix.update_asset_location(truck_id, loc[0], loc[1], node=node, timestamp=t)
                self.state_matrix.update_asset_status(truck_id, status)
                self.state_matrix.asset_metadata[row_idx]["fuel"] = fuel
            else:
                status_str = str(packet.get("status", "AVAILABLE")).upper()
                status = STATUS_BUSY if status_str == "BUSY" else STATUS_AVAILABLE
                self.state_matrix.register_asset(
                    asset_id=truck_id,
                    asset_type=ASSET_TYPE_TRUCK,
                    lat=loc[0],
                    lon=loc[1],
                    status=status,
                    capacity=capacity,
                    node=node,
                    fuel=fuel,
                    last_seen=t
                )
            return ("TRUCK_UPDATED", {"id": truck_id, "fuel": fuel, "status": status})

        return ("UNKNOWN_PACKET", packet)
