# RuleNet
# RuleNet: Rule-Augmented Neural Networks for Trustworthy Decision-Making

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A novel neuro-symbolic AI framework that integrates Datalog rules with deep neural networks to enhance explainability and trustworthiness in high-stakes decision-making domains like healthcare and finance.

## 📖 Abstract

Deep neural networks excel in predictive tasks but lack interpretability, hindering adoption in critical domains. RuleNet bridges symbolic logic and deep learning by augmenting DNNs with Datalog rules via predicate grounding and semantic loss. This ensures predictions align with domain constraints while maintaining predictive power, providing human-readable explanations for trustworthy AI.

## 🚀 Key Features

- **Neuro-Symbolic Integration**: Combines DNN flexibility with symbolic rule reasoning
- **Explainable Predictions**: Human-readable rule-based explanations
- **High-Stakes Domain Ready**: Validated on healthcare (MIMIC-III) and finance (Fraud-D) datasets
- **Efficient Inference**: ~4.5ms per prediction with 100% rule coverage
- **Scalable Architecture**: Lightweight rule augmentation without computational bottlenecks

## 📊 Performance Highlights

| Model | Accuracy | F1-Score | Rule Coverage | Inference Time |
|-------|----------|----------|---------------|----------------|
| MLP | 84.65% | 0.2878 | 0% | 5.2ms |
| CNN | 84.62% | 0.2764 | 0% | 7.8ms |
| **RuleNet** | **84.75%** | **0.2996** | **100%** | **4.5ms** |

## 🏗️ Architecture

### RuleNet Framework


Input Features → DNN Backbone → Predicate Grounding → Rule Satisfaction → Hybrid Prediction
↓ ↓ ↓
Data Learning + Symbolic Reasoning + Explainable Output


### Key Components
- **DNN Backbone**: Multi-layer perceptron for feature learning
- **Predicate Grounding**: Maps features to logical predicates
- **Semantic Loss**: Balances data fitting and rule adherence
- **Rule Engine**: Datalog-based constraint enforcement

## 📥 Installation


# Clone repository
git clone https://github.com/Farjana-Yesmin/RuleNet.git
cd RuleNet

# Install dependencies
pip install torch scikit-learn pandas numpy matplotlib imbalanced-learn kagglehub

🚀 Quick Start

from rulenet import RuleNet, RuleEngine
import torch

# Initialize model
input_dim = 5  # Number of features
hidden_dim = 128
model = RuleNet(input_dim, hidden_dim, output_dim=1)

# Define rules
rules = [
    lambda x: x[:, 0] > 1.0,  # High glucose
    lambda x: x[:, 2] > 1.5   # Obesity risk
]

# Train and evaluate
model.fit(X_train, y_train, rules=rules)
accuracy, f1, coverage = model.evaluate(X_test, y_test)

📁 Project Structure

RuleNet/
├── notebooks/
│   └── Northern_Lights_Deep_Learning.ipynb  # Main experiments
├── src/
│   ├── models/
│   │   ├── rulenet.py          # RuleNet implementation
│   │   ├── mlp.py              # MLP baseline
│   │   └── cnn.py              # CNN baseline
│   ├── data/
│   │   ├── preprocessing.py    # Data handling utilities
│   │   └── datasets.py         # Dataset loaders
│   └── utils/
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting utilities
├── data/                       # Dataset storage
├── results/                    # Experimental results
└── README.md

🗂️ Datasets

MIMIC-III (Healthcare)

Source: PhysioNet (requires access approval)
Features: Lab values, patient demographics, vital signs
Task: Diabetes prediction
Preprocessing: Synthetic augmentation for class balancing
Fraud-D (Financial)

Source: Kaggle (public)
Features: Transaction amount, balance changes, merchant info
Task: Fraud detection
Preprocessing: Standard scaling, SMOTE for imbalance


🔬 Experiments

Run the main notebook to reproduce experiments:

jupyter notebook notebooks/Northern_Lights_Deep_Learning.ipynb

Available Models

RuleNet: Our proposed rule-augmented architecture
MLP: Multi-layer perceptron baseline
CNN: Convolutional neural network (adapted for tabular data)
DeepProbLog: Neuro-symbolic baseline
KGReasoner: Knowledge graph reasoning baseline


📈 Results Visualization

The notebook includes comprehensive visualization:

Rule coverage across models
Performance metrics comparison
Inference time analysis
Ablation studies


🎯 Key Findings

Improved Trustworthiness: 100% rule coverage ensures predictions align with domain knowledge
Maintained Performance: Slight accuracy improvement over pure DNN approaches
Computational Efficiency: Faster inference than CNN baselines
Explainability: Gradient-based feature importance and rule satisfaction scores


📚 Citation

If you use RuleNet in your research, please cite:

@article{rulenet2024,
  title={Rule-Augmented Neural Networks for Trustworthy Decision-Making},
  author={Farjana Yesmin and Nusrat shirmin},
  journal={Conference on Neural Information Processing Systems},
  year={2024}
}


🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙋‍♂️ Support

For questions and support:

Open an issue on GitHub
Contact: farjanayesmin76@gmail.com


🏆 Acknowledgments

MIMIC-III dataset providers (PhysioNet)
Fraud-D dataset contributors (Kaggle)
Neuro-symbolic AI research community
RuleNet: Bridging the gap between neural performance and symbolic trustworthiness in AI systems.


