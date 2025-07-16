# Trustworthy Distributed Deep Learning: Adversarial Attack Mitigation in Model Parallelism

## Overview

This repository contains the implementation of a trustworthy distributed system for deep learning training under model parallelism, designed to mitigate adversarial attacks on large-scale models. Our approach focuses on maintaining model reliability and trustworthiness during distributed training by implementing trust scoring mechanisms and dynamic task reassignment.

## 🎯 Research Goals

- **Quantify adversarial attack effects** on large deep learning models across various datasets in distributed settings
- **Develop robust countermeasures** against adversarial attacks targeting model parameters and gradients
- **Implement trust scoring systems** for node reliability assessment
- **Create efficient task reassignment algorithms** for maintaining training continuity

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Security Layer │    │  Trust Layer    │
│                 │    │                 │    │                 │
│ • Data Loading  │    │ • Attack Det.   │    │ • Node Scoring  │
│ • Preprocessing │    │ • Gradient Ver. │    │ • Task Realloc. │
│ • Distribution  │    │ • Param Monitor │    │ • Trust Updates │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Training Engine │
                    │                 │
                    │ • Model Parallel│
                    │ • Gradient Agg. │
                    │ • Loss Tracking │
                    └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trustworthy-distributed-dl.git
cd trustworthy-distributed-dl

# Create conda environment
conda create -n trustworthy-dl python=3.8
conda activate trustworthy-dl

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from trustworthy_dl import DistributedTrainer, TrustManager
from trustworthy_dl.models import get_model
from trustworthy_dl.attacks import AdversarialAttacker

# Initialize distributed trainer
trainer = DistributedTrainer(
    model_name='gpt2',
    num_nodes=8,
    trust_threshold=0.7
)

# Setup trust management
trust_manager = TrustManager(
    initial_trust=0.9,
    decay_rate=0.1,
    recovery_rate=0.05
)

# Start training
trainer.train(
    dataset='openwebtext',
    epochs=10,
    trust_manager=trust_manager
)
```

## 📊 Supported Models

### Language Models
- GPT-2

### Computer Vision Models
- VGG (11, 13, 16)
- ResNet (32, 50, 101)



## 🔬 Experiments

### Running Experiments

```bash
# Image classification with adversarial attacks
python experiments/image_classification/run_cifar10_experiment.py

# Language modeling with model poisoning
python experiments/language_modeling/run_gpt2_experiment.py

```

### Configuration

All experiments can be configured via YAML files in the `configs/` directory:

```yaml
# configs/gpt2_distributed.yaml
model:
  name: "gpt2"
  size: "medium"
  
training:
  batch_size: 32
  learning_rate: 5e-5
  num_epochs: 10
  
distributed:
  num_nodes: 8
  parallelism: "model"
  
security:
  trust_threshold: 0.7
  attack_detection: true
  gradient_verification: true
```

## 📈 Expected Results

Based on our research objectives, you can expect to observe:

### 1. **Trust Score Dynamics**
- Trust scores decreasing for compromised nodes
- Recovery patterns for nodes after attack mitigation
- Correlation between trust scores and model performance

### 2. **Attack Detection Metrics**
- False positive/negative rates for different attack types
- Detection latency across various attack intensities
- Effectiveness of gradient verification techniques

### 3. **Model Performance Under Attack**
- Accuracy degradation patterns during adversarial attacks
- Loss function behavior with poisoned gradients
- Recovery time after node reassignment

### 4. **System Efficiency**
- Task reassignment overhead
- Training throughput with security measures
- Memory and computational overhead analysis

## 🧪 Evaluation Metrics

The system tracks multiple metrics during training:

- **Model Performance**: Accuracy, loss, perplexity
- **Security Metrics**: Attack detection rate, false positives
- **Trust Metrics**: Node trust scores, trust evolution
- **System Metrics**: Throughput, latency, resource utilization

## 🛠️ Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request


## 🏆 Publications

This work is intended for submission to:
- NSDI, OSDI, SOCC
- NDSS, CCS, ATC
- SIGCOMM, SOSP, EuroSys

## 👥 Team

- **Dr. Haiying Shen** - Principal Investigator, University of Virginia
- **Tanmoy Sen** - PostDoc, University of Virginia
- **Suraiya Tairin** - Ph.D. Student, University of Virginia


## 🙏 Acknowledgments

- DOE Argonne National Laboratory Testbed AI
- Delta GPU, NeoCortex AI resources
- LoneStar-6 GPU resources

## 📞 Contact

For questions or collaborations, please contact:
- Dr. Haiying Shen: [hs6ms@virginia.edu]

---

*This research is supported by computational resources from DOE national laboratories and aims to advance the security and trustworthiness of distributed deep learning systems.*
