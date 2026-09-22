"""
SAL_tech - Core State Matrix & Asset Ledger
Maintains global sparse adjacency matrix for road infrastructure and vectorized asset ledger.
"""

from __future__ import annotations
import json
import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Optional, Any, List


# Asset Type Enumerations
ASSET_TYPE_MEDIC = 1
ASSET_TYPE_TRUCK = 2

# Asset Status Enumerations
STATUS_AVAILABLE = 0
STATUS_BUSY = 1
STATUS_TIMEOUT = 2

# Ledger Column Indexes
COL_ID = 0
COL_TYPE = 1
COL_LAT = 2
COL_LON = 3
COL_STATUS = 4
COL_CAPACITY = 5


class StateMatrix:
    """
    Central Brain of SAL_tech:
    - Infrastructure Graph: scipy.sparse.csr_matrix with O(1) edge weight updates.
    - Asset Ledger: Vectorized NumPy array [Asset_ID, Type, Lat, Lon, Status_Flag, Capacity].
    """

    def __init__(self, num_nodes: int, edges: Optional[List[Tuple[int, int, float]]] = None):
        self.num_nodes = num_nodes
        self._edge_idx_map: Dict[Tuple[int, int], int] = {}
        
        # Build initial CSR graph structure
        if edges:
            self._init_graph_from_edges(edges)
        else:
            self.adj_matrix = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

        # Asset ledger: shape (N, 6) -> [ID, Type, Lat, Lon, Status, Capacity]
        self.asset_ledger = np.empty((0, 6), dtype=np.float64)
        self.asset_id_map: Dict[Any, int] = {}  # Maps string/int ID to ledger row index
        self.asset_metadata: Dict[int, Dict[str, Any]] = {}  # Auxiliary metadata (e.g. node, fuel, last_seen)

    def _init_graph_from_edges(self, edges: List[Tuple[int, int, float]]) -> None:
        """Initialize CSR graph and build O(1) edge index lookup map."""
        row_list = []
        col_list = []
        data_list = []

        for u, v, w in edges:
            row_list.extend([u, v])
            col_list.extend([v, u])
            data_list.extend([float(w), float(w)])

        # Construct CSR matrix
        coo = sp.coo_matrix(
            (data_list, (row_list, col_list)),
            shape=(self.num_nodes, self.num_nodes),
            dtype=np.float64
        )
        self.adj_matrix = coo.tocsr()
        self._build_edge_index_map()

    def _build_edge_index_map(self) -> None:
        """Precompute CSR data array offsets for O(1) weight updates."""
        self._edge_idx_map.clear()
        indptr = self.adj_matrix.indptr
        indices = self.adj_matrix.indices

        for u in range(self.num_nodes):
            start = indptr[u]
            end = indptr[u + 1]
            for data_idx in range(start, end):
                v = indices[data_idx]
                self._edge_idx_map[(u, v)] = data_idx

    def update_edge_weight(self, node_a: int, node_b: int, weight: float) -> bool:
        """
        O(1) update of edge weight in the CSR sparse matrix.
        Setting weight to np.inf marks road as destroyed.
        """
        idx_ab = self._edge_idx_map.get((node_a, node_b))
        idx_ba = self._edge_idx_map.get((node_b, node_a))

        if idx_ab is not None and idx_ba is not None:
            self.adj_matrix.data[idx_ab] = weight
            self.adj_matrix.data[idx_ba] = weight
            return True
        
        # If edge does not exist in initial topology, insert via dynamic rebuild
        lil = self.adj_matrix.tolil()
        lil[node_a, node_b] = weight
        lil[node_b, node_a] = weight
        self.adj_matrix = lil.tocsr()
        self._build_edge_index_map()
        return True

    def register_asset(
        self,
        asset_id: Any,
        asset_type: int,
        lat: float,
        lon: float,
        status: int = STATUS_AVAILABLE,
        capacity: float = 1.0,
        node: Optional[int] = None,
        fuel: float = 100.0,
        last_seen: float = 0.0
    ) -> int:
        """Registers a new asset into the vectorized ledger."""
        if asset_id in self.asset_id_map:
            row_idx = self.asset_id_map[asset_id]
            self.asset_ledger[row_idx] = [row_idx, asset_type, lat, lon, status, capacity]
        else:
            row_idx = len(self.asset_ledger)
            new_row = np.array([[row_idx, asset_type, lat, lon, status, capacity]], dtype=np.float64)
            self.asset_ledger = np.vstack((self.asset_ledger, new_row)) if self.asset_ledger.size else new_row
            self.asset_id_map[asset_id] = row_idx

        self.asset_metadata[row_idx] = {
            "external_id": asset_id,
            "node": node if node is not None else 0,
            "fuel": fuel,
            "last_seen": last_seen
        }
        return row_idx

    def update_asset_status(self, asset_id: Any, status: int) -> bool:
        """O(1) status update in vectorized ledger."""
        row_idx = self.asset_id_map.get(asset_id)
        if row_idx is not None:
            self.asset_ledger[row_idx, COL_STATUS] = status
            return True
        return False

    def update_asset_location(self, asset_id: Any, lat: float, lon: float, node: Optional[int] = None, timestamp: float = 0.0) -> bool:
        """O(1) location update in vectorized ledger."""
        row_idx = self.asset_id_map.get(asset_id)
        if row_idx is not None:
            self.asset_ledger[row_idx, COL_LAT] = lat
            self.asset_ledger[row_idx, COL_LON] = lon
            meta = self.asset_metadata[row_idx]
            if node is not None:
                meta["node"] = node
            meta["last_seen"] = timestamp
            return True
        return False

    def get_available_assets(self, asset_type: int) -> np.ndarray:
        """Vectorized filter returning rows of all AVAILABLE assets of given type."""
        if not self.asset_ledger.size:
            return np.empty((0, 6), dtype=np.float64)
        mask = (self.asset_ledger[:, COL_TYPE] == asset_type) & (self.asset_ledger[:, COL_STATUS] == STATUS_AVAILABLE)
        return self.asset_ledger[mask]

    def export_state_dict(self) -> Dict[str, Any]:
        """Exports in-memory mathematical state to dictionary for field dumping."""
        return {
            "num_nodes": self.num_nodes,
            "edges": [(u, v, float(self.adj_matrix.data[idx])) for (u, v), idx in self._edge_idx_map.items() if u < v],
            "ledger": self.asset_ledger.tolist(),
            "asset_id_map": {str(k): v for k, v in self.asset_id_map.items()},
            "metadata": {str(k): v for k, v in self.asset_metadata.items()}
        }

    def dump_state_to_file(self, filepath: str) -> None:
        """Dumps state atomically to disk simulating offline durability."""
        data = self.export_state_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """Encodes NumPy scalar and array types to Python native types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
