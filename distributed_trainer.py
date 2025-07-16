"""
Distributed Trainer for Trustworthy Deep Learning
Implements model parallelism with adversarial attack mitigation
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json
import os

from .trust_manager import TrustManager, NodeStatus
from .node_monitor import NodeMonitor
from ..security.gradient_verification import GradientVerifier
from ..security.attack_detection import AttackDetector
from ..utils.metrics import MetricsCollector
from ..models.model_factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingState(Enum):
    INITIALIZING = "initializing"
    TRAINING = "training"
    UNDER_ATTACK = "under_attack"
    RECOVERING = "recovering"
    COMPLETED = "completed"

@dataclass
class NodeConfig:
    """Configuration for individual nodes"""
    node_id: int
    rank: int
    world_size: int
    gpu_id: int
    model_partition: str
    trust_score: float = 1.0
    status: NodeStatus = NodeStatus.TRUSTED

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    dataset_name: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    num_nodes: int
    trust_threshold: float = 0.7
    attack_detection_enabled: bool = True
    gradient_verification_enabled: bool = True
    checkpoint_interval: int = 100
    max_reassignment_attempts: int = 3

class DistributedTrainer:
    """
    Main distributed training orchestrator with adversarial attack mitigation
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.training_state = TrainingState.INITIALIZING
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize components
        self.trust_manager = TrustManager(
            num_nodes=config.num_nodes,
            trust_threshold=config.trust_threshold
        )
        
        self.node_monitor = NodeMonitor()
        self.gradient_verifier = GradientVerifier()
        self.attack_detector = AttackDetector()
        self.metrics_collector = MetricsCollector()
        
        # Node configurations
        self.node_configs: Dict[int, NodeConfig] = {}
        self.model_partitions: Dict[int, torch.nn.Module] = {}
        
        # Training state
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        self.schedulers: Dict[int, torch.optim.lr_scheduler._LRScheduler] = {}
        
        # Attack tracking
        self.attack_history: List[Dict] = []
        self.reassignment_history: List[Dict] = []
        
        logger.info(f"Initialized DistributedTrainer with {config.num_nodes} nodes")

    def setup_distributed_environment(self, rank: int, world_size: int):
        """Setup distributed training environment"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        torch.cuda.set_device(rank)
        
        logger.info(f"Initialized distributed environment: rank {rank}/{world_size}")

    def create_model_partitions(self, model_name: str) -> Dict[int, torch.nn.Module]:
        """Create model partitions for distributed training"""
        model_factory = ModelFactory()
        base_model = model_factory.create_model(model_name)
        
        # Partition model across nodes
        partitions = {}
        
        if model_name.startswith('gpt'):
            # For GPT models, partition by layers
            layers_per_node = len(base_model.transformer.h) // self.config.num_nodes
            
            for node_id in range(self.config.num_nodes):
                start_layer = node_id * layers_per_node
                end_layer = min((node_id + 1) * layers_per_node, len(base_model.transformer.h))
                
                partition = torch.nn.Sequential(
                    *base_model.transformer.h[start_layer:end_layer]
                )
                partitions[node_id] = partition
                
        elif model_name.startswith('resnet'):
            # For ResNet models, partition by layer blocks
            # Implementation depends on specific architecture
            pass
            
        else:
            # Default partitioning strategy
            logger.warning(f"Using default partitioning for {model_name}")
            
        return partitions

    def forward_pass(self, inputs: torch.Tensor, node_sequence: List[int]) -> torch.Tensor:
        """Execute forward pass across distributed nodes"""
        current_output = inputs
        node_outputs = {}
        
        for node_id in node_sequence:
            if self.trust_manager.get_node_status(node_id) == NodeStatus.COMPROMISED:
                # Skip compromised nodes or use backup
                logger.warning(f"Skipping compromised node {node_id}")
                continue
                
            # Execute on current node
            partition = self.model_partitions[node_id]
            current_output = partition(current_output)
            
            # Monitor output for anomalies
            node_outputs[node_id] = current_output.clone()
            
            # Check for adversarial patterns
            if self.config.attack_detection_enabled:
                is_attack = self.attack_detector.detect_output_anomaly(
                    current_output, node_id, self.global_step
                )
                
                if is_attack:
                    self.handle_detected_attack(node_id, current_output)
                    
        return current_output, node_outputs

    def backward_pass(self, loss: torch.Tensor, node_sequence: List[int]) -> Dict[int, torch.Tensor]:
        """Execute backward pass with gradient verification"""
        gradients = {}
        
        # Compute gradients
        loss.backward()
        
        for node_id in reversed(node_sequence):
            if node_id not in self.model_partitions:
                continue
                
            partition = self.model_partitions[node_id]
            node_gradients = []
            
            for param in partition.parameters():
                if param.grad is not None:
                    node_gradients.append(param.grad.clone())
                    
            gradients[node_id] = node_gradients
            
            # Verify gradients for adversarial modifications
            if self.config.gradient_verification_enabled:
                is_valid = self.gradient_verifier.verify_gradients(
                    node_gradients, node_id, self.global_step
                )
                
                if not is_valid:
                    logger.warning(f"Invalid gradients detected from node {node_id}")
                    self.handle_gradient_attack(node_id, node_gradients)
                    
        return gradients

    def update_trust_scores(self, node_outputs: Dict[int, torch.Tensor], 
                          gradients: Dict[int, torch.Tensor]):
        """Update trust scores based on node behavior"""
        for node_id in range(self.config.num_nodes):
            # Calculate output deviation
            output_deviation = self.calculate_output_deviation(
                node_outputs.get(node_id), node_id
            )
            
            # Calculate gradient consistency
            gradient_consistency = self.calculate_gradient_consistency(
                gradients.get(node_id), node_id
            )
            
            # Update trust score
            self.trust_manager.update_trust_score(
                node_id, output_deviation, gradient_consistency
            )

    def calculate_output_deviation(self, output: torch.Tensor, node_id: int) -> float:
        """Calculate deviation of node output from expected range"""
        if output is None:
            return 1.0  # Maximum deviation for missing output
            
        # Get expected output statistics from historical data
        expected_mean = self.node_monitor.get_expected_mean(node_id)
        expected_std = self.node_monitor.get_expected_std(node_id)
        
        if expected_mean is None or expected_std is None:
            # Not enough historical data
            return 0.0
            
        # Calculate z-score based deviation
        actual_mean = output.mean().item()
        actual_std = output.std().item()
        
        mean_deviation = abs(actual_mean - expected_mean) / expected_std
        std_deviation = abs(actual_std - expected_std) / expected_std
        
        return min(1.0, (mean_deviation + std_deviation) / 2.0)

    def calculate_gradient_consistency(self, gradients: List[torch.Tensor], node_id: int) -> float:
        """Calculate gradient consistency score"""
        if not gradients:
            return 0.0
            
        # Check gradient norms
        grad_norms = [grad.norm().item() for grad in gradients]
        
        # Compare with expected gradient norms
        expected_norms = self.node_monitor.get_expected_gradient_norms(node_id)
        
        if not expected_norms:
            return 1.0  # No historical data, assume consistent
            
        # Calculate consistency score
        consistency_scores = []
        for norm, expected_norm in zip(grad_norms, expected_norms):
            if expected_norm > 0:
                consistency = min(1.0, norm / expected_norm)
                consistency_scores.append(consistency)
                
        return np.mean(consistency_scores) if consistency_scores else 1.0

    def handle_detected_attack(self, node_id: int, output: torch.Tensor):
        """Handle detected adversarial attack"""
        logger.error(f"Attack detected on node {node_id}")
        
        # Record attack
        attack_info = {
            'node_id': node_id,
            'timestamp': time.time(),
            'step': self.global_step,
            'attack_type': 'output_anomaly',
            'output_stats': {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'max': output.max().item(),
                'min': output.min().item()
            }
        }
        self.attack_history.append(attack_info)
        
        # Update trust score
        self.trust_manager.mark_compromised(node_id)
        
        # Trigger node reassignment
        self.reassign_node_tasks(node_id)
        
        # Update training state
        self.training_state = TrainingState.UNDER_ATTACK

    def handle_gradient_attack(self, node_id: int, gradients: List[torch.Tensor]):
        """Handle gradient-based attack"""
        logger.error(f"Gradient attack detected on node {node_id}")
        
        # Record attack
        attack_info = {
            'node_id': node_id,
            'timestamp': time.time(),
            'step': self.global_step,
            'attack_type': 'gradient_poisoning',
            'gradient_stats': {
                'norms': [grad.norm().item() for grad in gradients],
                'num_gradients': len(gradients)
            }
        }
        self.attack_history.append(attack_info)
        
        # Mark node as compromised
        self.trust_manager.mark_compromised(node_id)
        
        # Reassign tasks
        self.reassign_node_tasks(node_id)

    def reassign_node_tasks(self, compromised_node_id: int):
        """Reassign tasks from compromised node to trusted nodes"""
        logger.info(f"Reassigning tasks from node {compromised_node_id}")
        
        # Find best replacement node
        trusted_nodes = self.trust_manager.get_trusted_nodes()
        
        if not trusted_nodes:
            logger.error("No trusted nodes available for reassignment")
            return
            
        # Select node with highest trust score and lowest load
        best_node = max(trusted_nodes, key=lambda n: self.trust_manager.get_trust_score(n))
        
        # Estimate migration time
        migration_time = self.estimate_migration_time(compromised_node_id, best_node)
        
        # Perform reassignment
        self.perform_task_reassignment(compromised_node_id, best_node)
        
        # Record reassignment
        reassignment_info = {
            'from_node': compromised_node_id,
            'to_node': best_node,
            'timestamp': time.time(),
            'migration_time': migration_time,
            'step': self.global_step
        }
        self.reassignment_history.append(reassignment_info)

    def estimate_migration_time(self, source_node: int, target_node: int) -> float:
        """Estimate time required for task migration"""
        # Simple estimation based on model partition size
        partition_size = sum(p.numel() for p in self.model_partitions[source_node].parameters())
        
        # Assume 1GB/s transfer rate
        transfer_time = partition_size * 4 / (1024**3)  # 4 bytes per float32
        
        # Add setup overhead
        setup_time = 2.0  # 2 seconds
        
        return transfer_time + setup_time

    def perform_task_reassignment(self, source_node: int, target_node: int):
        """Perform actual task reassignment"""
        # Move model partition
        source_partition = self.model_partitions[source_node]
        
        # For now, duplicate the partition on the target node
        # In a real implementation, this would involve more complex load balancing
        if target_node not in self.model_partitions:
            self.model_partitions[target_node] = source_partition
            
        # Update node configuration
        self.node_configs[target_node].model_partition = f"partition_{source_node}"
        
        logger.info(f"Task reassignment completed: {source_node} -> {target_node}")

    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch"""
        self.current_epoch = epoch
        epoch_loss = 0.0
        num_batches = 0
        
        # Define node sequence for forward pass
        node_sequence = list(range(self.config.num_nodes))
        
        for batch_idx, batch in enumerate(dataloader):
            self.global_step += 1
            
            # Forward pass
            outputs, node_outputs = self.forward_pass(batch['input'], node_sequence)
            
            # Calculate loss
            loss = self.calculate_loss(outputs, batch['target'])
            
            # Backward pass
            gradients = self.backward_pass(loss, node_sequence)
            
            # Update trust scores
            self.update_trust_scores(node_outputs, gradients)
            
            # Optimizer step
            self.optimizer_step(gradients)
            
            # Collect metrics
            batch_metrics = {
                'loss': loss.item(),
                'step': self.global_step,
                'epoch': epoch,
                'trust_scores': {i: self.trust_manager.get_trust_score(i) 
                               for i in range(self.config.num_nodes)}
            }
            self.metrics_collector.collect_batch_metrics(batch_metrics)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Check for checkpointing
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss

    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate training loss"""
        # Implement based on specific model and task
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, targets)

    def optimizer_step(self, gradients: Dict[int, torch.Tensor]):
        """Perform optimizer step with gradient aggregation"""
        for node_id, node_gradients in gradients.items():
            if node_id in self.optimizers:
                self.optimizers[node_id].step()
                self.optimizers[node_id].zero_grad()

    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_partitions': {k: v.state_dict() for k, v in self.model_partitions.items()},
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'trust_scores': {i: self.trust_manager.get_trust_score(i) 
                           for i in range(self.config.num_nodes)},
            'attack_history': self.attack_history,
            'reassignment_history': self.reassignment_history
        }
        
        checkpoint_path = f"checkpoints/checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
            
        logger.info(f"Starting training for {num_epochs} epochs")
        self.training_state = TrainingState.TRAINING
        
        for epoch in range(num_epochs):
            # Training phase
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            # Validation phase
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
            # Check training state
            if self.training_state == TrainingState.UNDER_ATTACK:
                logger.info("Training under attack - implementing recovery measures")
                self.training_state = TrainingState.RECOVERING
                
            # Update learning rate
            for scheduler in self.schedulers.values():
                scheduler.step()
                
        self.training_state = TrainingState.COMPLETED
        logger.info("Training completed successfully")

    def validate(self, val_dataloader) -> float:
        """Validation phase"""
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                node_sequence = list(range(self.config.num_nodes))
                outputs, _ = self.forward_pass(batch['input'], node_sequence)
                loss = self.calculate_loss(outputs, batch['target'])
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches

    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'training_state': self.training_state.value,
            'trust_scores': {i: self.trust_manager.get_trust_score(i) 
                           for i in range(self.config.num_nodes)},
            'attack_count': len(self.attack_history),
            'reassignment_count': len(self.reassignment_history),
            'metrics': self.metrics_collector.get_summary()
        }

    def cleanup(self):
        """Cleanup distributed training resources"""
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("Distributed training cleanup completed")
