"""
Trust Manager for Node Reliability Assessment
Implements dynamic trust scoring and node status management
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    TRUSTED = "trusted"
    SUSPICIOUS = "suspicious"
    COMPROMISED = "compromised"
    RECOVERING = "recovering"
    OFFLINE = "offline"

@dataclass
class TrustScore:
    """Trust score with metadata"""
    value: float
    last_updated: float
    update_count: int
    decay_rate: float = 0.01
    recovery_rate: float = 0.005

@dataclass
class NodeMetrics:
    """Comprehensive node metrics for trust calculation"""
    output_deviation: float = 0.0
    gradient_consistency: float = 1.0
    communication_latency: float = 0.0
    resource_utilization: float = 0.0
    error_rate: float = 0.0
    uptime: float = 1.0

class TrustManager:
    """
    Manages trust scores and node status for distributed training
    """
    
    def __init__(self, num_nodes: int, trust_threshold: float = 0.7,
                 initial_trust: float = 1.0, max_history: int = 1000):
        self.num_nodes = num_nodes
        self.trust_threshold = trust_threshold
        self.initial_trust = initial_trust
        self.max_history = max_history
        
        # Initialize trust scores
        self.trust_scores: Dict[int, TrustScore] = {}
        self.node_status: Dict[int, NodeStatus] = {}
        self.node_metrics: Dict[int, NodeMetrics] = {}
        
        # Historical data
        self.trust_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.attack_history: Dict[int, List] = defaultdict(list)
        self.performance_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
        # Weights for trust calculation
        self.trust_weights = {
            'output_deviation': 0.3,
            'gradient_consistency': 0.3,
            'communication_latency': 0.1,
            'resource_utilization': 0.1,
            'error_rate': 0.15,
            'uptime': 0.05
        }
        
        # Initialize all nodes
        for node_id in range(num_nodes):
            self.initialize_node(node_id)
            
        logger.info(f"TrustManager initialized for {num_nodes} nodes")

    def initialize_node(self, node_id: int):
        """Initialize a node with default trust values"""
        self.trust_scores[node_id] = TrustScore(
            value=self.initial_trust,
            last_updated=time.time(),
            update_count=0
        )
        self.node_status[node_id] = NodeStatus.TRUSTED
        self.node_metrics[node_id] = NodeMetrics()

    def update_trust_score(self, node_id: int, output_deviation: float, 
                          gradient_consistency: float, **kwargs):
        """Update trust score based on node behavior"""
        if node_id not in self.trust_scores:
            self.initialize_node(node_id)
            
        # Update node metrics
        metrics = self.node_metrics[node_id]
        metrics.output_deviation = output_deviation
        metrics.gradient_consistency = gradient_consistency
        
        # Update additional metrics if provided
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
                
        # Calculate new trust score
        new_trust = self._calculate_trust_score(node_id, metrics)
        
        # Apply temporal decay
        old_trust = self.trust_scores[node_id]
        time_diff = time.time() - old_trust.last_updated
        decay_factor = np.exp(-old_trust.decay_rate * time_diff)
        
        # Weighted combination of old and new trust
        alpha = 0.1  # Learning rate
        final_trust = (1 - alpha) * old_trust.value * decay_factor + alpha * new_trust
        final_trust = np.clip(final_trust, 0.0, 1.0)
        
        # Update trust score
        self.trust_scores[node_id] = TrustScore(
            value=final_trust,
            last_updated=time.time(),
            update_count=old_trust.update_count + 1,
            decay_rate=old_trust.decay_rate,
            recovery_rate=old_trust.recovery_rate
        )
        
        # Update node status based on trust score
        self._update_node_status(node_id, final_trust)
        
        # Record in history
        self.trust_history[node_id].append({
            'timestamp': time.time(),
            'trust_score': final_trust,
            'metrics': metrics.__dict__.copy()
        })
        
        logger.debug(f"Node {node_id} trust updated: {final_trust:.3f}")

    def _calculate_trust_score(self, node_id: int, metrics: NodeMetrics) -> float:
        """Calculate trust score based on weighted metrics"""
        # Convert metrics to trust components (higher is better)
        trust_components = {
            'output_deviation': 1.0 - min(1.0, metrics.output_deviation),
            'gradient_consistency': metrics.gradient_consistency,
            'communication_latency': 1.0 - min(1.0, metrics.communication_latency / 10.0),
            'resource_utilization': min(1.0, metrics.resource_utilization),
            'error_rate': 1.0 - min(1.0, metrics.error_rate),
            'uptime': metrics.uptime
        }
        
        # Weighted sum
        trust_score = sum(
            self.trust_weights[component] * value
            for component, value in trust_components.items()
        )
        
        return np.clip(trust_score, 0.0, 1.0)

    def _update_node_status(self, node_id: int, trust_score: float):
        """Update node status based on trust score"""
        current_status = self.node_status[node_id]
        
        if trust_score < 0.3:
            new_status = NodeStatus.COMPROMISED
        elif trust_score < self.trust_threshold:
            new_status = NodeStatus.SUSPICIOUS
        elif current_status == NodeStatus.COMPROMISED and trust_score > 0.8:
            new_status = NodeStatus.RECOVERING
        elif current_status == NodeStatus.RECOVERING and trust_score > 0.9:
            new_status = NodeStatus.TRUSTED
        elif trust_score >= self.trust_threshold:
            new_status = NodeStatus.TRUSTED
        else:
            new_status = current_status
            
        if new_status != current_status:
            logger.info(f"Node {node_id} status changed: {current_status.value} -> {new_status.value}")
            self.node_status[node_id] = new_status

    def mark_compromised(self, node_id: int, attack_type: str = "unknown"):
        """Mark a node as compromised due to detected attack"""
        self.node_status[node_id] = NodeStatus.COMPROMISED
        self.trust_scores[node_id].value = 0.1  # Severe trust penalty
        
        # Record attack
        attack_record = {
            'timestamp': time.time(),
            'attack_type': attack_type,
            'previous_trust': self.trust_scores[node_id].value
        }
        self.attack_history[node_id].append(attack_record)
        
        logger.warning(f"Node {node_id} marked as compromised: {attack_type}")

    def initiate_recovery(self, node_id: int):
        """Initiate recovery process for a compromised node"""
        if self.node_status[node_id] == NodeStatus.COMPROMISED:
            self.node_status[node_id] = NodeStatus.RECOVERING
            
            # Increase recovery rate temporarily
            self.trust_scores[node_id].recovery_rate = 0.02
            
            logger.info(f"Recovery initiated for node {node_id}")

    def get_trust_score(self, node_id: int) -> float:
        """Get current trust score for a node"""
        if node_id not in self.trust_scores:
            return 0.0
        return self.trust_scores[node_id].value

    def get_node_status(self, node_id: int) -> NodeStatus:
        """Get current status for a node"""
        return self.node_status.get(node_id, NodeStatus.OFFLINE)

    def get_trusted_nodes(self) -> List[int]:
        """Get list of currently trusted nodes"""
        return [
            node_id for node_id in range(self.num_nodes)
            if self.node_status[node_id] == NodeStatus.TRUSTED
        ]

    def get_suspicious_nodes(self) -> List[int]:
        """Get list of suspicious nodes"""
        return [
            node_id for node_id in range(self.num_nodes)
            if self.node_status[node_id] == NodeStatus.SUSPICIOUS
        ]

    def get_compromised_nodes(self) -> List[int]:
        """Get list of compromised nodes"""
        return [
            node_id for node_id in range(self.num_nodes)
            if self.node_status[node_id] == NodeStatus.COMPROMISED
        ]

    def can_assign_task(self, node_id: int) -> bool:
        """Check if a task can be assigned to a node"""
        status = self.node_status.get(node_id, NodeStatus.OFFLINE)
        return status in [NodeStatus.TRUSTED, NodeStatus.RECOVERING]

    def select_best_nodes(self, num_nodes: int) -> List[int]:
        """Select the best nodes for task assignment"""
        # Get available nodes with their trust scores
        available_nodes = [
            (node_id, self.get_trust_score(node_id))
            for node_id in range(self.num_nodes)
            if self.can_assign_task(node_id)
        ]
        
        # Sort by trust score (descending)
        available_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Return top nodes
        return [node_id for node_id, _ in available_nodes[:num_nodes]]

    def calculate_system_trust(self) -> float:
        """Calculate overall system trust level"""
        if not self.trust_scores:
            return 0.0
            
        trust_values = [score.value for score in self.trust_scores.values()]
        
        # Weighted average (give more weight to higher trust scores)
        weights = np.array(trust_values)
        weighted_trust = np.average(trust_values, weights=weights)
        
        return weighted_trust

    def get_trust_statistics(self) -> Dict:
        """Get comprehensive trust statistics"""
        trust_values = [score.value for score in self.trust_scores.values()]
        
        if not trust_values:
            return {}
            
        stats = {
            'mean_trust': np.mean(trust_values),
            'std_trust': np.std(trust_values),
            'min_trust': np.min(trust_values),
            'max_trust': np.max(trust_values),
            'system_trust': self.calculate_system_trust(),
            'node_status_counts': {
                status.value: sum(1 for s in self.node_status.values() if s == status)
                for status in NodeStatus
            },
            'total_attacks': sum(len(attacks) for attacks in self.attack_history.values())
        }
        
        return stats

    def get_node_history(self, node_id: int, limit: int = 100) -> List[Dict]:
        """Get trust history for a specific node"""
        if node_id not in self.trust_history:
            return []
            
        history = list(self.trust_history[node_id])
        return history[-limit:] if limit else history

    def export_trust_data(self, filepath: str):
        """Export trust data to JSON file"""
        export_data = {
            'trust_scores': {
                str(node_id): {
                    'value': score.value,
                    'last_updated': score.last_updated,
                    'update_count': score.update_count
                }
                for node_id, score in self.trust_scores.items()
            },
            'node_status': {
                str(node_id): status.value
                for node_id, status in self.node_status.items()
            },
            'trust_history': {
                str(node_id): list(history)
                for node_id, history in self.trust_history.items()
            },
            'attack_history': {
                str(node_id): attacks
                for node_id, attacks in self.attack_history.items()
            },
            'statistics': self.get_trust_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Trust data exported to {filepath}")

    def adaptive_threshold_adjustment(self):
        """Dynamically adjust trust threshold based on system state"""
        stats = self.get_trust_statistics()
        mean_trust = stats.get('mean_trust', 0.7)
        
        # If overall trust is low, temporarily lower threshold
        if mean_trust < 0.5:
            self.trust_threshold = max(0.3, mean_trust - 0.1)
        elif mean_trust > 0.9:
            self.trust_threshold = min(0.8, mean_trust - 0.1)
        else:
            # Gradually return to default
            default_threshold = 0.7
            self.trust_threshold += 0.01 * (default_threshold - self.trust_threshold)
            
        logger.debug(f"Trust threshold adjusted to {self.trust_threshold:.3f}")

    def predict_node_reliability(self, node_id: int, horizon: int = 10) -> float:
        """Predict future reliability of a node"""
        if node_id not in self.trust_history or len(self.trust_history[node_id]) < 5:
            return self.get_trust_score(node_id)
            
        # Simple trend analysis
        recent_scores = [
            entry['trust_score'] 
            for entry in list(self.trust_history[node_id])[-10:]
        ]
        
        # Calculate trend
        x = np.arange(len(recent_scores))
        coeffs = np.polyfit(x, recent_scores, 1)
        
        # Predict future score
        future_score = coeffs[0] * (len(recent_scores) + horizon) + coeffs[1]
        
        return np.clip(future_score, 0.0, 1.0)

    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving system trust"""
        recommendations = []
        stats = self.get_trust_statistics()
        
        if stats.get('mean_trust', 1.0) < 0.6:
            recommendations.append("System trust is low - consider investigating compromised nodes")
            
        compromised_nodes = self.get_compromised_nodes()
        if len(compromised_nodes) > self.num_nodes * 0.3:
            recommendations.append("High number of compromised nodes - check security measures")
            
        if stats.get('total_attacks', 0) > 10:
            recommendations.append("Frequent attacks detected - strengthen attack detection")
            
        suspicious_nodes = self.get_suspicious_nodes()
        if suspicious_nodes:
            recommendations.append(f"Monitor suspicious nodes: {suspicious_nodes}")
            
        return recommendations

    def reset_node_trust(self, node_id: int):
        """Reset a node's trust to initial value"""
        self.initialize_node(node_id)
        logger.info(f"Trust reset for node {node_id}")

    def cleanup(self):
        """Cleanup trust manager resources"""
        logger.info("TrustManager cleanup completed")