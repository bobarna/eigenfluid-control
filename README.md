# Controlling Laplacian Eigenfluids
## Scientific Students' Association Report 2022
### Author: Barnabás Börcsök
### Advisor: dr. László Szécsi

- [thesis.pdf](https://bobarna.github.io/eigenfluid-control/thesis.pdf)
    - Automatically deployed from `docs` folder
- For an informal overview of some basic concepts, see [this blog post](https://barnabasborcsok.com/posts/eigenfluid-control/).

## Abstract
Understanding and modelling our environment is a great and important challenge, span- ning many disciplines from weather and climate forecast, through vehicle design to com- puter graphics. Physical systems are usually described by Partial Differential Equations (PDEs), which we can approximate using established numerical techniques. Next to pre- dicting outcomes, planning interactions to control physical systems is also a long-standing problem.

In our work, we investigate the use of Laplacian Eigenfunctions to model and control fluid flow. We make use of an explicit description of our simulation domain to derive gradients of the physical simulation, enabling neural network agents to learn to control the physical process to achieve desired outcomes.

## Running the Code

### Dependencies
- Python
- [Jupyter notebook](https://jupyter.org/install)
- [ΦFlow](https://github.com/tum-pbs/PhiFlow)
    - dash
- [Pytorch](https://pytorch.org/)
- [Tensorflow](https://www.tensorflow.org/) -- as an alternative backend

All of these can be installed via [pip](https://pypi.org/project/pip/) on [Python 3.6](https://www.python.org/downloads/) and above:
```
pip install phiflow dash torch torchvision tensorflow
pip install notebook
```

#### Using [conda](https://docs.conda.io/en/latest/)
As an alternative to [pip](https://pypi.org/project/pip/), `environment.yml` describes a [conda](https://docs.conda.io/en/latest/) environemt for installing every necessary Python package in a managed environment.

Create conda environment based on `environment.yml`:
```
conda env create
```
Activate the conda environment:
```
conda activate eigenfluid-control
```

### Running the notebooks
```
jupyter notebook
```

*TODO* describe layout of code base
