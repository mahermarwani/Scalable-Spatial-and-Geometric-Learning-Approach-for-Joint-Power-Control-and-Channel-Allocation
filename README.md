
---

# 📡 Scalable Spatial and Geometric Learning for Joint Power Control & Channel Allocation

Welcome to the official repository for:

> **Scalable Spatial and Geometric Learning Approach for Joint Power Control and Channel Allocation**  
> *By Maher Marwani & Georges Kaddoum*  
> Published in *IEEE Transactions on Wireless Communications*, 2024

---

## 📚 Table of Contents

1. [🔬 Overview](#-overview)  
2. [📊 Results & Visualizations](#-results--visualizations)  
   - [📍 Simulation Setup](#-simulation-setup)  
   - [📈 Performance Visualization](#-performance-visualization)  
3. [📄 Citation](#-citation)  
4. [🧠 Abstract](#-abstract)  
5. [⚙️ Installation & Usage](#️-installation--usage)  
   - [🔧 Setup](#-setup)  
   - [🚀 Running the Code](#-running-the-code)  
6. [📬 Questions or Feedback?](#-questions-or-feedback)

---

## 🔬 Overview

This project presents a **scalable, geometry-aware deep learning framework** that addresses the **joint power control and spectrum allocation** problem in wireless networks. Specifically tailored for **Device-to-Device (D2D)** communications, our approach leverages a combination of **Variational Autoencoders (VAE)**, **Convolutional Neural Networks (CNNs)**, and **Graph Neural Networks (GNNs)** to effectively model the non-Euclidean nature of wireless topologies.

Conventional resource allocation techniques struggle with scalability and geometric generalization. This work overcomes those limitations using deep geometric learning, enabling intelligent and **adaptive resource management** in highly dynamic interference environments.

---

## 📊 Results & Visualizations

### 📍 Simulation Setup  
- **Deployment Area**: 200m × 100m  
- **Network Topology**: 20 D2D communication links  
- **Minimum Data Rate Constraint**: 500 bps  
- **Bandwidth**: 5 resource blocks, each 500 Hz wide

### 📈 Performance Visualization  
The animation below illustrates the **evolution of link-level data rates** over time:

![Rate Animation](results/rate_animation.gif)

This visualization highlights the model’s ability to **adaptively manage interference and resource constraints** through learned spatial and topological features.

---

## 📄 Citation

If this work contributes to your research or applications, please cite us:

```bibtex
@ARTICLE{10667016,
  author={Marwani, Maher and Kaddoum, Georges},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Scalable Spatial and Geometric Learning Approach for Joint Power Control and Channel Allocation}, 
  year={2024},
  volume={23},
  number={11},
  pages={16976-16991},
  keywords={Resource management;Device-to-device communication;Power control;Interference;Graph neural networks;Channel allocation;Heuristic algorithms;Intelligent resource allocation;RRM;6G;D2D;GNN;CNN;VAE},
  doi={10.1109/TWC.2024.3449036}
}
```

---

## 🧠 Abstract

We introduce a **scalable, unsupervised, probabilistic learning framework** for joint spectrum and power allocation in D2D networks — a crucial component of future 6G wireless systems. While previous DL-based approaches suffer from scalability issues and inadequate modeling of spatial relationships, our method integrates CNNs, GNNs, and VAEs to extract and preserve **spatial and geometric features** from raw channel state information (CSI).

Key innovations include:  
- A **hybrid CNN-GNN-VAE architecture** tailored to capture both **topological and spatial correlations**  
- An **attention-based GNN module** that refines the network’s understanding of critical link dependencies  
- A **generalizable solution** that performs across diverse environments without retraining or architecture redesign

Extensive experiments validate the method’s **robust performance and adaptability**, setting a new benchmark for intelligent wireless resource management.

---

## ⚙️ Installation & Usage

To get started:

### 🔧 Setup

1. **Clone this Repository**
   ```bash
   git clone https://github.com/mahermarwani/Scalable-Spatial-and-Geometric-Learning-Approach-for-Joint-Power-Control-and-Channel-Allocation.git
   cd Scalable-Spatial-and-Geometric-Learning-Approach-for-Joint-Power-Control-and-Channel-Allocation
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

### 🚀 Running the Code

1. **Generate the Dataset**
   ```bash
   python data_generation.py
   ```

2. **Use the Complete Model (VAE + CNN + GNN)**
   ```bash
   python final_model.py
   ```

3. **Train the GNN Embedding Model (LRGAT)**
   ```bash
   python Geometric_embedding.py
   ```

4. **Train the CNN Embedding Model**
   ```bash
   python Spacial_embedding.py
   ```

5. **Explore Simulation Options and Comparison Baselines**  
   Additional scripts are included for testing heuristic baselines and visualizing comparative performance.

---

## 📬 Questions or Feedback?

We welcome your input! Open an [issue](https://github.com/mahermarwani/Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation/issues) or submit a [pull request](https://github.com/mahermarwani/Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation/pulls) to contribute. 

---