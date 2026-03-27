# State Space Models Mamba and the Selective Scan

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

## Overview

Transformers scale as O(L²) prohibitive for long sequences. **State Space Models (SSMs)** offer O(L log L) training and O(1) inference memory. **Mamba** (Gu & Dao, 2023) adds input-dependent selectivity, making it the primary transformer competitor for sequence modelling.

**Technique:** S4, Mamba, selective scan, HiPPO, ZOH discretisation  
**Year:** S4: 2022, Mamba: 2023 current frontier  
**Difficulty:** Advanced / beyond-course control theory meets deep learning

## What You Will Learn
- SSM equations from Kalman (1960) control theory
- ZOH discretisation continuous (A,B) → discrete (Ā,B̄)
- S4 with HiPPO initialisation
- Mamba's selective scan input-dependent B, C, Δ
- Dual computation: CNN training (parallel) + RNN inference (O(1) memory)
- Complete S4 and Mamba implementation in PyTorch

## Quick Start
```bash
git clone https://github.com/yourusername/ssm-tutorial.git
cd ssm-tutorial
pip install torch numpy matplotlib scikit-learn notebook
jupyter notebook ssm_tutorial.ipynb
```

## References
1. Gu et al. (2022) S4. https://arxiv.org/abs/2111.00396
2. Gu & Dao (2023) Mamba. https://arxiv.org/abs/2312.00752
3. Kalman (1960) State space models. https://doi.org/10.1115/1.3662552
4. Dao & Gu (2024) Transformers are SSMs. https://arxiv.org/abs/2405.21060

