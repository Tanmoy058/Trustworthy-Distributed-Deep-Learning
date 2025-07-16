"""
Attack Detection System for Distributed Deep Learning
Implements various attack detection mechanisms for data and model poisoning
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque
import time
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class AttackType(Enum):
    DATA_POISONING = "data_poisoning"
    MODEL_POISONING = "model_poisoning"
    GRADIENT_POISONING = "gradient_poisoning"
    BYZANTINE = "byzantine"
    BACKDOOR = "backdoor"
    ADVERSARIAL_INPUT = "adversarial_input"

@dataclass
class AttackDetectionResult:
    """Result of attack detection"""
    is_attack: bool
    attack_type: Optional[AttackType]
    confidence: float
    evidence: Dict[str, Any]
    timestamp: float
    node_id: int

class AttackDetector:
    """
    Comprehensive attack detection system for distributed training
    """
    
    def __init__(self, detection_threshold: float = 0.8, 
                 history_size: int = 1000):
        self.detection_threshold = detection_threshold
        self.history_size = history_size
        
        # Historical data for baseline establishment
        self.output_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.gradient_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.loss_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Statistical baselines
        self.output_baselines: Dict[int, Dict] = defaultdict(dict)
        self.gradient_baselines: Dict[int, Dict] = defaultdict(dict)
        
        # Attack detection models
        self.anomaly_detectors: Dict[int, IsolationForest] = {}
        self.clustering_models: Dict[int, DBSCAN] = {}
        
        # Detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'attack_types': defaultdict(int)
        }
        
        logger.info("AttackDetector initialized")

    def detect_output_anomaly(self, output: torch.Tensor, node_id: int, 
                            step: int) -> bool:
        """Detect anomalies in node output"""
        if output is None:
            return True
            
        # Convert to numpy for analysis
        output_np = output.detach().cpu().numpy().flatten()
        
        # Statistical analysis
        output_stats = self._calculate_tensor_statistics(output_np)
        
        # Update history
        self.output_history[node_id].append({
            'step': step,
            'stats': output_stats,
            'timestamp': time.time()
        })
        
        # Check if we have enough history for detection
        if len(self.output_history[node_id]) < 10:
            return False
            
        # Update baseline
        self._update_output_baseline(node_id)
        
        # Perform detection
        detection_result = self._detect_statistical_anomaly(
            output_stats, self.output_baselines[node_id], node_id
        )
        
        if detection_result.is_attack:
            logger.warning(f"Output anomaly detected on node {node_id}: {detection_result.attack_type}")
            self.detection_stats['total_detections'] += 1
            self.detection_stats['attack_types'][detection_result.attack_type.value] += 1
            
        return detection_result.is_attack

    def detect_gradient_poisoning(self, gradients: List[torch.Tensor], 
                                node_id: int, step: int) -> bool:
        """Detect gradient poisoning attacks"""
        if not gradients:
            return False
            
        # Calculate gradient statistics
        grad_stats = self._calculate_gradient_statistics(gradients)
        
        # Update history
        self.gradient_history[node_id].append({
            'step': step,
            'stats': grad_stats,
            'timestamp': time.time()
        })
        
        # Check for sufficient history
        if len(self.gradient_history[node_id]) < 10:
            return False
            
        # Update baseline
        self._update_gradient_baseline(node_id)
        
        # Detect anomalies
        detection_result = self._detect_gradient_anomaly(
            grad_stats, self.gradient_baselines[node_id], node_id
        )
        
        if detection_result.is_attack:
            logger.warning(f"Gradient poisoning detected on node {node_id}")
            self.detection_stats['total_detections'] += 1
            
        return detection_result.is_attack

    def detect_byzantine_behavior(self, node_outputs: Dict[int, torch.Tensor], 
                                 step: int) -> List[int]:
        """Detect Byzantine failures through cross-node comparison"""
        if len(node_outputs) < 3:
            return []
            
        byzantine_nodes = []
        
        # Calculate pairwise similarities
        similarities = self._calculate_output_similarities(node_outputs)
        
        # Identify outliers
        for node_id, similarity_scores in similarities.items():
            avg_similarity = np.mean(list(similarity_scores.values()))
            
            if avg_similarity < 0.5:  # Threshold for Byzantine detection
                byzantine_nodes.append(node_id)
                logger.warning(f"Byzantine behavior detected on node {node_id}")
                
        return byzantine_nodes

    def detect_backdoor_attack(self, model_outputs: torch.Tensor, 
                             expected_outputs: torch.Tensor, 
                             node_id: int) -> bool:
        """Detect backdoor attacks through output analysis"""
        if model_outputs is None or expected_outputs is None:
            return False
            
        # Calculate output divergence
        divergence = torch.nn.functional.kl_div(
            torch.log_softmax(model_outputs, dim=-1),
            torch.softmax(expected_outputs, dim=-1),
            reduction='batchmean'
        )
        
        # Check if divergence exceeds threshold
        if divergence.item() > 2.0:  # Threshold for backdoor detection
            logger.warning(f"Potential backdoor attack detected on node {node_id}")
            return True
            
        return False

    def _calculate_tensor_statistics(self, tensor: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics for a tensor"""
        return {
            'mean': float(np.mean(tensor)),
            'std': float(np.std(tensor)),
            'min': float(np.min(tensor)),
            'max': float(np.max(tensor)),
            'median': float(np.median(tensor)),
            'skewness': float(stats.skew(tensor)),
            'kurtosis': float(stats.kurtosis(tensor)),
            'percentile_25': float(np.percentile(tensor, 25)),
            'percentile_75': float(np.percentile(tensor, 75)),
            'norm_l1': float(np.linalg.norm(tensor, ord=1)),
            'norm_l2': float(np.linalg.norm(tensor, ord=2)),
            'norm_inf': float(np.linalg.norm(tensor, ord=np.inf))
        }

    def _calculate_gradient_statistics(self, gradients: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate statistics for gradients"""
        if not gradients:
            return {}
            
        # Flatten all gradients
        all_grads = torch.cat([g.flatten() for g in gradients])
        grad_np = all_grads.detach().cpu().numpy()
        
        # Calculate norms for each gradient tensor
        grad_norms = [g.norm().item() for g in gradients]
        
        stats_dict = self._calculate_tensor_statistics(grad_np)
        stats_dict.update({
            'num_gradients': len(gradients),
            'grad_norms_mean': float(np.mean(grad_norms)),
            'grad_norms_std': float(np.std(grad_norms)),
            'grad_norms_max': float(np.max(grad_norms)),
            'cosine_similarity': self._calculate_gradient_cosine_similarity(gradients)
        })
        
        return stats_dict

    def _calculate_gradient_cosine_similarity(self, gradients: List[torch.Tensor]) -> float:
        """Calculate average cosine similarity between gradients"""
        if len(gradients) < 2:
            return 1.0
            
        similarities = []
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                sim = torch.nn.functional.cosine_similarity(
                    gradients[i].flatten().unsqueeze(0),
                    gradients[j].flatten().unsqueeze(0)
                )
                similarities.append(sim.item())
                
        return float(np.mean(similarities)) if similarities else 1.0

    def _update_output_baseline(self, node_id: int):
        """Update statistical baseline for node outputs"""
        history = list(self.output_history[node_id])
        
        if len(history) < 10:
            return
            
        # Aggregate statistics
        aggregated_stats = defaultdict(list)
        for entry in history:
            for stat_name, stat_value in entry['stats'].items():
                aggregated_stats[stat_name].append(stat_value)
                
        # Calculate baseline statistics
        baseline = {}
        for stat_name, values in aggregated_stats.items():
            baseline[stat_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_95': np.percentile(values, 95)
            }
            
        self.output_baselines[node_id] = baseline

    def _update_gradient_baseline(self, node_id: int):
        """Update statistical baseline for node gradients"""
        history = list(self.gradient_history[node_id])
        
        if len(history) < 10:
            return
            
        # Similar to output baseline update
        aggregated_stats = defaultdict(list)
        for entry in history:
            for stat_name, stat_value in entry['stats'].items():
                aggregated_stats[stat_name].append(stat_value)
                
        baseline = {}
        for stat_name, values in aggregated_stats.items():
            baseline[stat_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_95': np.percentile(values, 95)
            }
            
        self.gradient_baselines[node_id] = baseline

    def _detect_statistical_anomaly(self, current_stats: Dict[str, float], 
                                   baseline: Dict[str, Dict], 
                                   node_id: int) -> AttackDetectionResult:
        """Detect statistical anomalies using baseline comparison"""
        if not baseline:
            return AttackDetectionResult(
                is_attack=False,
                attack_type=None,
                confidence=0.0,
                evidence={},
                timestamp=time.time(),
                node_id=node_id
            )
            
        anomaly_scores = []
        evidence = {}
        
        for stat_name, current_value in current_stats.items():
            if stat_name not in baseline:
                continue
                
            base_stats = baseline[stat_name]
            
            # Z-score based anomaly detection
            if base_stats['std'] > 0:
                z_score = abs((current_value - base_stats['mean']) / base_stats['std'])
                anomaly_scores.append(z_score)
                
                if z_score > 3:  # 3-sigma rule
                    evidence[stat_name] = {
                        'z_score': z_score,
                        'current_value': current_value,
                        'baseline_mean': base_stats['mean'],
                        'baseline_std': base_stats['std']
                    }
                    
        # Overall anomaly score
        overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        is_anomaly = overall_score > 2.5  # Threshold for anomaly
        
        # Determine attack type based on evidence
        attack_type = self._classify_attack_type(evidence, current_stats)
        
        return AttackDetectionResult(
            is_attack=is_anomaly,
            attack_type=attack_type if is_anomaly else None,
            confidence=min(1.0, overall_score / 5.0),
            evidence=evidence,
            timestamp=time.time(),
            node_id=node_id
        )

    def _detect_gradient_anomaly(self, grad_stats: Dict[str, float],
                               baseline: Dict[str, Dict],
                               node_id: int) -> AttackDetectionResult:
        """Detect gradient-specific anomalies"""
        return self._detect_statistical_anomaly(grad_stats, baseline, node_id)

    def _classify_attack_type(self, evidence: Dict, stats: Dict) -> Optional[AttackType]:
        """Classify the type of attack based on evidence"""
        if not evidence:
            return None
            
        # Simple rule-based classification
        if 'norm_l2' in evidence and evidence['norm_l2']['z_score'] > 5:
            return AttackType.GRADIENT_POISONING
        elif 'std' in evidence and evidence['std']['z_score'] > 4:
            return AttackType.DATA_POISONING
        elif 'skewness' in evidence or 'kurtosis' in evidence:
            return AttackType.ADVERSARIAL_INPUT
        else:
            return AttackType.BYZANTINE

    def _calculate_output_similarities(self, node_outputs: Dict[int, torch.Tensor]) -> Dict[int, Dict[int, float]]:
        """Calculate similarities between node outputs"""
        similarities = defaultdict(dict)
        
        for node1, output1 in node_outputs.items():
            for node2, output2 in node_outputs.items():
                if node1 != node2:
                    # Calculate cosine similarity
                    sim = torch.nn.functional.cosine_similarity(
                        output1.flatten().unsqueeze(0),
                        output2.flatten().unsqueeze(0)
                    )
                    similarities[node1][node2] = sim.item()
                    
        return dict(similarities)

    def update_detection_models(self):
        """Update machine learning models for anomaly detection"""
        for node_id in self.output_history.keys():
            if len(self.output_history[node_id]) < 50:
                continue
                
            # Prepare training data
            features = []
            for entry in self.output_history[node_id]:
                feature_vector = list(entry['stats'].values())
                features.append(feature_vector)
                
            features = np.array(features)
            
            # Train isolation forest for anomaly detection
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(features)
            self.anomaly_detectors[node_id] = iso_forest
            
            # Train DBSCAN for clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(features)
            self.clustering_models[node_id] = dbscan
            
        logger.info("Detection models updated")

    def detect_with_ml_models(self, stats: Dict[str, float], node_id: int) -> bool:
        """Use trained ML models for attack detection"""
        if node_id not in self.anomaly_detectors:
            return False
            
        feature_vector = np.array(list(stats.values())).reshape(1, -1)
        
        # Use isolation forest
        anomaly_score = self.anomaly_detectors[node_id].decision_function(feature_vector)[0]
        is_anomaly = self.anomaly_detectors[node_id].predict(feature_vector)[0] == -1
        
        if is_anomaly:
            logger.debug(f"ML model detected anomaly on node {node_id}, score: {anomaly_score}")
            
        return is_anomaly

    def get_detection_statistics(self) -> Dict:
        """Get comprehensive detection statistics"""
        total_detections = self.detection_stats['total_detections']
        
        stats = {
            'total_detections': total_detections,
            'false_positive_rate': self.detection_stats['false_positives'] / max(1, total_detections),
            'true_positive_rate': self.detection_stats['true_positives'] / max(1, total_detections),
            'attack_type_distribution': dict(self.detection_stats['attack_types']),
            'nodes_monitored': len(self.output_history),
            'average_history_length': np.mean([len(h) for h in self.output_history.values()]) if self.output_history else 0
        }
        
        return stats

    def set_detection_threshold(self, threshold: float):
        """Update detection threshold"""
        self.detection_threshold = np.clip(threshold, 0.0, 1.0)
        logger.info(f"Detection threshold updated to {self.detection_threshold}")

    def reset_node_history(self, node_id: int):
        """Reset detection history for a specific node"""
        if node_id in self.output_history:
            self.output_history[node_id].clear()
        if node_id in self.gradient_history:
            self.gradient_history[node_id].clear()
        if node_id in self.output_baselines:
            del self.output_baselines[node_id]
        if node_id in self.gradient_baselines:
            del self.gradient_baselines[node_id]
            
        logger.info(f"Detection history reset for node {node_id}")

    def export_detection_data(self, filepath: str):
        """Export detection data for analysis"""
        export_data = {
            'detection_stats': self.detection_stats,
            'baselines': {
                'output': {str(k): v for k, v in self.output_baselines.items()},
                'gradient': {str(k): v for k, v in self.gradient_baselines.items()}
            },
            'history_lengths': {
                str(node_id): len(history) 
                for node_id, history in self.output_history.items()
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Detection data exported to {filepath}")

    def cleanup(self):
        """Cleanup detector resources"""
        self.output_history.clear()
        self.gradient_history.clear()
        self.loss_history.clear()
        self.anomaly_detectors.clear()
        self.clustering_models.clear()
        logger.info("AttackDetector cleanup completed")