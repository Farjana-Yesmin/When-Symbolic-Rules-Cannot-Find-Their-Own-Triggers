# RuleNet
# Joint Denoising and 3D Point Cloud Reconstruction from Single Medical Images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A preliminary joint learning framework for simultaneous medical image denoising and 3D point cloud reconstruction from single noisy images.

## 📋 Overview

This repository contains the implementation of a compact two-branch neural network that performs **joint denoising and 3D reconstruction** from single medical images. The model learns shared intermediate representations to efficiently handle both tasks within a single network, offering a resource-efficient alternative to sequential pipelines.

**Key Features:**
- 🏗️ Two-branch architecture with shared representations
- 🔄 Simultaneous image denoising and 3D point cloud reconstruction
- 💡 Designed for resource-constrained environments (Google Colab compatible)
- 🧪 Proof-of-concept implementation with synthetic data

## 🚀 Quick Start

### Prerequisites


pip install torch torchvision transformers numpy matplotlib scipy scikit-learn plyfile SimpleITK

Usage

Clone the repository:

git clone https://github.com/farjana-yesmin/joint-denoising-3dreconstruction.git
cd joint-denoising-3dreconstruction
Run the Jupyter notebook:

jupyter notebook Joint_Denoising_and_3D_Point_Cloud_Reconstruction_from_Single_Medical_Images.ipynb
Or execute directly in Google Colab:

Upload the notebook to Google Colab
Ensure GPU runtime is selected (Runtime → Change runtime type → GPU)

🏗️ Model Architecture

The model employs a dual-branch architecture:

Denoising Branch: Convolutional layers with batch normalization and dropout (0.3)
Reconstruction Branch: Linear layer projecting to 1000 3D point coordinates
Shared Representations: Intermediate features shared between both tasks
text
Input (128×128) → Shared Encoder → [Denoising Head | Reconstruction Head]
Loss Function

The total loss combines pixel-level and geometric constraints:


L_total = λ_denoise · L_MSE + λ_recon · L_Chamfer
where λ_denoise = 1.0, λ_recon = 1.0

📊 Results

Quantitative Performance (Synthetic Data)

Metric	Sequential Baseline	Joint Framework
PSNR (dB)	10.00 ± 0.50	10.00 ± 0.50
Chamfer Distance	0.1086 ± 0.0050	0.0535 ± 0.0050
SSIM	0.4601 ± 0.0100	0.3370 ± 0.0100
Qualitative Results

denoising_comparison.png
point_cloud.png

🗂️ Dataset

Intended Dataset: LIDC-IDRI (Lung Image Database Consortium)

Status: Limited access in current implementation
Current Solution: Synthetic noisy 2D slices with Gaussian noise (σ=0.1)
Preprocessing: Images resized to 128×128, point clouds normalized to [-1,1]
Data Split: 70% training, 20% validation, 10% testing

⚙️ Training Configuration


# Hyperparameters
learning_rate = 1e-5
batch_size = 1 (with gradient accumulation over 4 steps)
epochs = 50
optimizer = Adam
loss_weights = [1.0, 1.0]  # denoising, reconstruction

Hardware Requirements:

Minimum: Google Colab T4 GPU (fallback to CPU supported)
Recommended: GPU with ≥8GB VRAM for full LIDC-IDRI experiments


🎯 Key Findings


✅ Feasibility demonstrated: Joint learning is possible even under constrained resources

✅ Geometric fidelity: Chamfer Distance improved by ~50% compared to sequential baseline

⚠️ Denoising challenge: Modest PSNR/SSIM due to synthetic data limitations

💡 Clinical potential: Coarse 3D previews achievable from noisy single images




🔮 Future Work

Real-data validation on complete LIDC-IDRI dataset
Efficiency benchmarking against sequential pipelines
Advanced architectures (diffusion models, transformers)
Multi-modal extension (CT, MRI, X-ray)
Clinical workflow integration

📝 Citation

If you use this work in your research, please cite:

bibtex
@article{anonymous2024joint,
  title={Joint Denoising and 3D Point Cloud Reconstruction from Single Medical Images},
  author={Farjana Yesmin and Nusrat Shirmin},
  journal={Preprint},
  year={2024}
}

👥 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

Google Colab for computational resources
LIDC-IDRI dataset providers
The open-source medical imaging community

Note: This is a proof-of-concept implementation. Results may vary with different datasets and hardware configurations.
