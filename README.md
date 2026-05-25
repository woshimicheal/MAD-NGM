# MAD-NGM: Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations
Sample code for Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations with Deep Networks


## File Directory Structure
```text
MAD-NGM/
├── Data/                    # Small demonstration datasets used to test the code workflow
├── MAD/                     # Implementation of the Meta-Auto-Decoder (MAD) module
├── NG/                      # JAX-based deep neural network modules for MAD combined with NG/RSNG
├── ops/                     # Computational modules for Neural Galerkin method and Random sparse Neural Galerkin scheme
├── kdv_mad_ngm.py           # Main script for solving the Korteweg--de Vries equation
├── burgers_mad_ngm.py       # Main script for solving the Burgers equation
├── ac1d_mad_ngm.py          # Main script for solving the one-dimensional Allen--Cahn equation
├── ac2d_mad_ngm.py          # Main script for solving the two-dimensional Allen--Cahn equation
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation

## Installation
Then install jax with the appropriate CPU or GPU support: [here](https://github.com/google/jax#installation)

Install all additionaly required packages run:

```bash
 pip install -r requirements.txt
```

## Citing this work
Qiuqi Li, Yiting Liu, Jin Zhao, Wencan Zhu, "MAD-NG: Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations". arXiv preprint arXiv:2512.21633 (2025).

```bash
@misc{li2025madngmetaautodecoderneuralgalerkin,
      title={MAD-NG: Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations}, 
      author={Qiuqi Li and Yiting Liu and Jin Zhao and Wencan Zhu},
      year={2025},
      eprint={2512.21633},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.21633}, 
}
```
