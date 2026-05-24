# MAD-NGM: Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations
Sample code for Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations with Deep Networks


## Repository Structure
- Data/:  Contains small demonstration datasets used to test the code workflow. 
- MAD/: Contains the implementation of Meta Auto Decoder (MAD).
- NG/: Contains the DNN network implemented based on the JAX framework for MAD combined with NG/RSNG.
- ops/: Contains Computational module for the Neural Galerkin method (NG/RSNG)

- kdv_mad_ngm.py: for solving the Korteweg-de Vries equation .
- burgers_mad_ngm.py: for solving the Burgers equation .
- ac1d_mad_ngm.py: for solving the one-dimensional Allen--Cahn equation .
- ac2d_mad_ngm.py: for solving the two-dimensional Allen--Cahn equation .

- requirements.txt: List of required packages to run the code.
- README.md: This file.


## Installation
Then install jax with the appropriate CPU or GPU support: [here](https://github.com/google/jax#installation)
Install all additionaly required packages run:

```bash
 pip install -r requirements.txt
```
