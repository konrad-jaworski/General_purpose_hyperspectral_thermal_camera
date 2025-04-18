# General Purpose Hyperspectral Thermal Camera

This repository contains tools and modules for simulating, optimizing, and analyzing hyperspectral thermal imaging systems. It encompasses legacy approaches, current developments, and utilities for thin-film filter simulations.

## Repository Structure

### `Theory_MK_1/` and `Theory_MK_2/`

These directories house legacy optimization approaches based on earlier theoretical models. They are preserved to document the progression and milestones achieved in the project's development.

### `Minimal_number_of_filters/`

This module explores the impact of varying the number of filters on system performance. It currently reflects results from previous optimization strategies and is slated for updates incorporating findings from the ongoing Lorentzian approach, which utilizes continuous function bases for optimization.

### `torch_tmm/`

A PyTorch-based library designed for simulating thin-film filters using the Transfer Matrix Method (TMM). Leveraging PyTorch's capabilities, it offers:

- GPU-accelerated computations
- Analytical gradient calculations via Autograd
- Vectorized operations for efficient simulations

This facilitates rapid and scalable simulations essential for optimizing multilayer thin-film structures.

### `utils_optimization/`

A collection of utility functions supporting the optimization process, including:

- Simulation setup and attribute configuration
- Data handling routines for saving and retrieving simulation results
- Auxiliary functions aiding in the optimization workflow

## Current Development: `Lorentzian_function_approach/`

The project is currently advancing towards an optimization methodology that employs continuous function bases, specifically utilizing Lorentzian functions. This approach aims to enhance the precision and adaptability of filter design within hyperspectral thermal imaging systems.

## Getting Started

To utilize the modules and tools within this repository:

1. Clone the repository:

   ```bash
   git clone https://github.com/konrad-jaworski/General_purpose_hyperspectral_thermal_camera.git
