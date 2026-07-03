# 🧠 Deep Learning for PDEs — TUM Exam Project

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red?logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

My solutions for the final project of the course  
**"Deep Learning for PDEs in Engineering Physics"** at the **Technical University of Munich**.

> Four different deep learning methods were implemented **entirely from scratch** using PyTorch — no pre‑built PDE libraries (DeepXDE, neuraloperator, etc.) were used.

---

## 📊 Results at a Glance

| Problem | Method | L² Relative Error |
|:--------|:-------|:-----------------|
| **A** – Inverse recovery of Young’s modulus | PINN + trigonometric regularisation | **u:** 0.58 % **k:** 3.35 % |
| **B** – Darcy flow in heterogeneous media | Deep Ritz method (hard BC) | 2.96 % |
| **C** – Heat conduction surrogate | Fourier Neural Operator (FNO) | 3.1 % |
| **D** – Burgers’ traffic flow prediction | Supervised DeepONet | 5.87 % |

*All metrics are relative L² errors on the official test set. Convergence plots and logs are available inside each project folder.*

---

## 📂 Repository Structure
