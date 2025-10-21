"""
Graph Neural Network service for fraud detection and relationship analysis.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict, deque
import json
from dataclasses import dataclass

from src.config.settings import get_settings
from src.database.repositories import transaction_repo
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class GraphNode:
    """Represents a node in the transaction graph."""
    node_id: str
    node_type: str  # 'account', 'device', 'merchant', 'location'
    features: Dict[str, float]
    risk_score: float = 0.0
    last_updated: datetime = None


@dataclass
class GraphEdge:
    """Represents an edge in the transaction graph."""
    source: str
    target: str
    edge_type: str  # 'transaction', 'device_shared', 'location_shared'
    weight: float = 1.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None


class TransactionGraph:
    """Transaction graph for fraud analysis."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.node_types = {'account', 'device', 'merchant', 'location'}
        self.edge_types = {'transaction', 'device_shared', 'location_shared', 'merchant_shared'}
        
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = set()
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        edge_key = (edge.source, edge.target)
        self.edges[edge_key] = edge
        self.adjacency[edge.source].add(edge.target)
        self.adjacency[edge.target].add(edge.source)
    
    def get_neighbors(self, node_id: str, max_depth: int = 2) -> Set[str]:
        """Get neighbors within max_depth hops."""
        visited = set()
        current_level = {node_id}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                if node not in visited:
                    visited.add(node)
                    next_level.update(self.adjacency[node])
            current_level = next_level - visited
        
        return visited - {node_id}
    
    def get_suspicious_paths(self, node_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find suspicious paths from a node."""
        suspicious_paths = []
        
        def dfs_path(current_path: List[str], depth: int):
            if depth > max_depth:
                return
            
            current_node = current_path[-1]
            neighbors = self.adjacency[current_node]
            
            for neighbor in neighbors:
                if neighbor not in current_path:  # Avoid cycles
                    new_path = current_path + [neighbor]
                    
                    # Calculate path suspiciousness
                    path_suspiciousness = self._calculate_path_suspiciousness(new_path)
                    
                    if path_suspiciousness > 0.5:  # Threshold for suspicious paths
                        suspicious_paths.append({
                            "path": new_path,
                            "suspiciousness": path_suspiciousness,
                            "depth": len(new_path) - 1,
                            "node_types": [self.nodes[node].node_type for node in new_path]
                        })
                    
                    if depth < max_depth:
                        dfs_path(new_path, depth + 1)
        
        dfs_path([node_id], 0)
        
        # Sort by suspiciousness and return top paths
        suspicious_paths.sort(key=lambda x: x["suspiciousness"], reverse=True)
        return suspicious_paths[:10]  # Top 10 suspicious paths
    
    def _calculate_path_suspiciousness(self, path: List[str]) -> float:
        """Calculate suspiciousness score for a path."""
        if len(path) < 2:
            return 0.0
        
        total_risk = 0.0
        edge_weights = 0.0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            
            # Node risk scores
            if source in self.nodes:
                total_risk += self.nodes[source].risk_score
            if target in self.nodes:
                total_risk += self.nodes[target].risk_score
            
            # Edge weight
            edge_key = (source, target)
            if edge_key in self.edges:
                edge_weights += self.edges[edge_key].weight
        
        # Normalize by path length
        avg_risk = total_risk / (len(path) * 2)  # Each node counted twice
        avg_edge_weight = edge_weights / max(len(path) - 1, 1)
        
        return min(avg_risk * avg_edge_weight, 1.0)


class GNNFraudService:
    """Graph Neural Network service for fraud detection."""
    
    def __init__(self):
        self.graph = TransactionGraph()
        self.is_initialized = False
        self.update_interval = 300  # 5 minutes
        self.max_graph_size = 10000  # Maximum nodes in memory
        self.last_update = None
        
    async def initialize(self) -> None:
        """Initialize the GNN service."""
        try:
            await self._build_initial_graph()
            self.is_initialized = True
            logger.info("GNN Fraud Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GNN service: {e}")
            raise
    
    async def _build_initial_graph(self) -> None:
        """Build initial transaction graph from recent transactions."""
        try:
            # Get recent transactions (last 7 days)
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            # This would be replaced with actual database query
            # For now, we'll create a sample graph
            await self._create_sample_graph()
            
            logger.info(f"Initial graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error building initial graph: {e}")
            raise
    
    async def _create_sample_graph(self) -> None:
        """Create a sample graph for testing."""
        # Sample accounts
        accounts = [
            "acc_123456", "acc_789012", "acc_345678", "acc_901234",
            "acc_567890", "acc_111111", "acc_222222", "acc_333333"
        ]
        
        # Sample devices
        devices = [
            "dev_001", "dev_002", "dev_003", "dev_004", "dev_005"
        ]
        
        # Sample merchants
        merchants = [
            "merchant_001", "merchant_002", "merchant_003"
        ]
        
        # Add account nodes
        for acc in accounts:
            node = GraphNode(
                node_id=acc,
                node_type="account",
                features={
                    "balance": np.random.uniform(100, 10000),
                    "age_days": np.random.uniform(30, 3650),
                    "transaction_count_30d": np.random.poisson(20)
                },
                risk_score=np.random.uniform(0.1, 0.9),
                last_updated=datetime.utcnow()
            )
            self.graph.add_node(node)
        
        # Add device nodes
        for dev in devices:
            node = GraphNode(
                node_id=dev,
                node_type="device",
                features={
                    "device_age_days": np.random.uniform(1, 365),
                    "transaction_count": np.random.poisson(50),
                    "risk_score": np.random.uniform(0.2, 0.8)
                },
                risk_score=np.random.uniform(0.2, 0.8),
                last_updated=datetime.utcnow()
            )
            self.graph.add_node(node)
        
        # Add merchant nodes
        for mer in merchants:
            node = GraphNode(
                node_id=mer,
                node_type="merchant",
                features={
                    "merchant_category": np.random.choice(["retail", "services", "utilities"]),
                    "transaction_volume": np.random.uniform(1000, 50000),
                    "chargeback_rate": np.random.uniform(0.01, 0.05)
                },
                risk_score=np.random.uniform(0.1, 0.7),
                last_updated=datetime.utcnow()
            )
            self.graph.add_node(node)
        
        # Add transaction edges
        transaction_pairs = [
            ("acc_123456", "acc_789012"),
            ("acc_123456", "acc_345678"),
            ("acc_789012", "acc_901234"),
            ("acc_345678", "acc_567890"),
            ("acc_901234", "acc_111111"),
            ("acc_567890", "acc_222222"),
            ("acc_111111", "acc_333333"),
        ]
        
        for source, target in transaction_pairs:
            edge = GraphEdge(
                source=source,
                target=target,
                edge_type="transaction",
                weight=np.random.uniform(0.5, 1.0),
                timestamp=datetime.utcnow(),
                metadata={"amount": np.random.uniform(10, 1000)}
            )
            self.graph.add_edge(edge)
        
        # Add device sharing edges
        device_sharing = [
            ("acc_123456", "dev_001"),
            ("acc_789012", "dev_001"),  # Shared device
            ("acc_345678", "dev_002"),
            ("acc_901234", "dev_002"),  # Shared device
            ("acc_567890", "dev_003"),
            ("acc_111111", "dev_004"),
            ("acc_222222", "dev_005"),
            ("acc_333333", "dev_005"),  # Shared device
        ]
        
        for account, device in device_sharing:
            edge = GraphEdge(
                source=account,
                target=device,
                edge_type="device_shared",
                weight=1.0,
                timestamp=datetime.utcnow()
            )
            self.graph.add_edge(edge)
    
    async def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction using graph-based features."""
        try:
            sender_account = transaction_data.get("sender_account")
            receiver_account = transaction_data.get("receiver_account")
            
            if not sender_account or not receiver_account:
                return {"graph_risk_score": 0.5, "graph_evidence": None}
            
            # Get graph-based risk scores
            sender_risk = self._get_node_risk_score(sender_account)
            receiver_risk = self._get_node_risk_score(receiver_account)
            
            # Calculate relationship risk
            relationship_risk = await self._calculate_relationship_risk(sender_account, receiver_account)
            
            # Get suspicious paths
            suspicious_paths = self.graph.get_suspicious_paths(sender_account, max_depth=3)
            
            # Calculate overall graph risk
            graph_risk_score = (sender_risk + receiver_risk + relationship_risk) / 3
            
            # Prepare graph evidence
            graph_evidence = []
            for path in suspicious_paths[:3]:  # Top 3 suspicious paths
                graph_evidence.append({
                    "path": path["path"],
                    "suspiciousness": path["suspiciousness"],
                    "node_types": path["node_types"]
                })
            
            return {
                "graph_risk_score": float(graph_risk_score),
                "sender_risk": float(sender_risk),
                "receiver_risk": float(receiver_risk),
                "relationship_risk": float(relationship_risk),
                "graph_evidence": graph_evidence,
                "neighbor_count": len(self.graph.adjacency.get(sender_account, set())),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transaction with GNN: {e}")
            return {"graph_risk_score": 0.5, "graph_evidence": None}
    
    def _get_node_risk_score(self, node_id: str) -> float:
        """Get risk score for a node."""
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id].risk_score
        return 0.5  # Default neutral risk
    
    async def _calculate_relationship_risk(self, sender: str, receiver: str) -> float:
        """Calculate relationship risk between two accounts."""
        try:
            # Check if accounts are connected
            if receiver in self.graph.adjacency.get(sender, set()):
                # Direct connection - check edge weight
                edge_key = (sender, receiver)
                if edge_key in self.graph.edges:
                    edge = self.graph.edges[edge_key]
                    return min(edge.weight, 1.0)
            
            # Check for shared connections (common neighbors)
            sender_neighbors = self.graph.adjacency.get(sender, set())
            receiver_neighbors = self.graph.adjacency.get(receiver, set())
            common_neighbors = sender_neighbors.intersection(receiver_neighbors)
            
            if common_neighbors:
                # Higher risk if many common neighbors
                shared_risk = min(len(common_neighbors) * 0.1, 0.8)
                return shared_risk
            
            return 0.3  # Default relationship risk
            
        except Exception as e:
            logger.error(f"Error calculating relationship risk: {e}")
            return 0.5
    
    async def update_graph_with_transaction(self, transaction_data: Dict[str, Any]) -> None:
        """Update graph with new transaction."""
        try:
            sender_account = transaction_data.get("sender_account")
            receiver_account = transaction_data.get("receiver_account")
            
            if not sender_account or not receiver_account:
                return
            
            # Add/update nodes
            await self._update_or_create_node(sender_account, "account", transaction_data)
            await self._update_or_create_node(receiver_account, "account", transaction_data)
            
            # Add transaction edge
            edge = GraphEdge(
                source=sender_account,
                target=receiver_account,
                edge_type="transaction",
                weight=1.0,
                timestamp=datetime.utcnow(),
                metadata={
                    "amount": transaction_data.get("amount", 0),
                    "channel": transaction_data.get("channel", "unknown")
                }
            )
            self.graph.add_edge(edge)
            
            # Update risk scores based on transaction
            await self._update_risk_scores(sender_account, receiver_account, transaction_data)
            
            logger.debug(f"Updated graph with transaction {transaction_data.get('transaction_id')}")
            
        except Exception as e:
            logger.error(f"Error updating graph with transaction: {e}")
    
    async def _update_or_create_node(self, node_id: str, node_type: str, transaction_data: Dict[str, Any]) -> None:
        """Update existing node or create new one."""
        if node_id in self.graph.nodes:
            # Update existing node
            node = self.graph.nodes[node_id]
            node.last_updated = datetime.utcnow()
            # Update features based on transaction
            if "transaction_count_30d" in node.features:
                node.features["transaction_count_30d"] += 1
        else:
            # Create new node
            node = GraphNode(
                node_id=node_id,
                node_type=node_type,
                features={
                    "balance": transaction_data.get("amount", 0),
                    "transaction_count_30d": 1,
                    "first_seen": datetime.utcnow()
                },
                risk_score=0.5,  # Default risk
                last_updated=datetime.utcnow()
            )
            self.graph.add_node(node)
    
    async def _update_risk_scores(self, sender: str, receiver: str, transaction_data: Dict[str, Any]) -> None:
        """Update risk scores based on transaction patterns."""
        try:
            amount = transaction_data.get("amount", 0)
            
            # Update sender risk (higher amounts = higher risk)
            if sender in self.graph.nodes:
                sender_node = self.graph.nodes[sender]
                amount_risk = min(amount / 10000, 0.5)  # Normalize amount risk
                sender_node.risk_score = min(sender_node.risk_score + amount_risk * 0.1, 1.0)
            
            # Update receiver risk
            if receiver in self.graph.nodes:
                receiver_node = self.graph.nodes[receiver]
                amount_risk = min(amount / 10000, 0.5)
                receiver_node.risk_score = min(receiver_node.risk_score + amount_risk * 0.05, 1.0)
            
        except Exception as e:
            logger.error(f"Error updating risk scores: {e}")
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "node_types": {
                node_type: sum(1 for node in self.graph.nodes.values() if node.node_type == node_type)
                for node_type in self.graph.node_types
            },
            "edge_types": {
                edge_type: sum(1 for edge in self.graph.edges.values() if edge.edge_type == edge_type)
                for edge_type in self.graph.edge_types
            },
            "avg_risk_score": np.mean([node.risk_score for node in self.graph.nodes.values()]) if self.graph.nodes else 0.0,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
    
    async def cleanup_old_data(self) -> None:
        """Clean up old graph data to prevent memory issues."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            
            # Remove old edges
            old_edges = [
                edge_key for edge_key, edge in self.graph.edges.items()
                if edge.timestamp and edge.timestamp < cutoff_time
            ]
            
            for edge_key in old_edges:
                source, target = edge_key
                self.graph.adjacency[source].discard(target)
                self.graph.adjacency[target].discard(source)
                del self.graph.edges[edge_key]
            
            logger.info(f"Cleaned up {len(old_edges)} old edges")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


# Global GNN service instance
gnn_service = GNNFraudService()
