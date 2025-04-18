# General_purpose_hyperspectral_thermal_camera

This repository contains tools and modules for simulating, optimizing, and analyzing hyperspectral thermal imaging systems. It encompasses legacy approaches, current developments, and utilities for thin-film filter simulations.​

Repository Structure:

Theory_MK_1/ and Theory_MK_2/

  These directories house legacy optimization approaches based on earlier theoretical models. They are preserved to document the progression and milestones achieved in the project's development.​
  FasterCapital

Minimal_number_of_filters/

  This module explores the impact of varying the number of filters on system performance. It currently reflects results from previous optimization strategies and is slated for updates incorporating findings from the ongoing   Lorentzian approach, which utilizes continuous function bases for optimization.​

torch_tmm/

  A PyTorch-based library designed for simulating thin-film filters using the Transfer Matrix Method (TMM). Leveraging PyTorch's capabilities.

utils_optimization/

  A collection of utility functions supporting the optimization process, including:​
    - Simulation setup and attribute configuration
    - Data handling routines for saving and retrieving simulation results
    - Auxiliary functions aiding in the optimization workflow​


# Current Development: Lorentzian Approach
Newest approach which utilizes Lorentzian as function basis to guide thin layer filter design.

# Getting Started
To utilize the modules and tools within this repository:​

Clone the repository:​




License
This project is licensed under the MIT License.
