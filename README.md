# Controlling Laplacian Eigenfluids
## Scientific Students' Association Report 2022
### Author: Barnabás Börcsök
### Advisor: dr. László Szécsi 

- [thesis.pdf](https://bobarna.github.io/eigenfluid-control/thesis.pdf)
    - Automatically deployed from `docs` folder
- For an informal overview of some basic concepts, see [this blog post](https://barnabasborcsok.com/posts/eigenfluid-control/).

## Abstract
Understanding and modeling our environment is a great and important challenge,
spanning many disciplines from weather and climate forecast, through vehicle
design to computer graphics. Physical systems are usually described by Partial
Differential Equations (PDEs), which we can approximate using established
numerical techniques. Next to predicting outcomes, planning interactions to
control physical systems is also a longstanding problem.

In our work, we investigate the use of Laplacian Eigenfunctions to model and
control fluid flow. We make use of an explicit description of our simulation
domain to derive gradients of the physical simulation, enabling neural network
agents to learn to control the physical process to achieve desired outcomes.

## Running the Code

### Dependencies
- [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow)
    - dash
- [Pytorch](https://pytorch.org/)
- [Tensorflow](https://www.tensorflow.org/) -- as an alternative backend
- [Jupyter notebook](https://jupyter.org/install) -- for running the notebooks

All of these can be installed via [pip](https://pypi.org/project/pip/) on
[Python 3.6](https://www.python.org/downloads/) and above:
```
pip install phiflow==2.2.2 dash torch torchvision tensorflow
pip install notebook
```

### Running the Notebooks
#### Locally
```
jupyter notebook
```
#### In Google Colab
- [Intro](https://colab.research.google.com/github/bobarna/eigenfluid-control/blob/main/eigenfluid_intro.ipynb)
- [Optimization -- Velocity Only](https://colab.research.google.com/github/bobarna/eigenfluid-control/blob/main/eigenfluid-optimization-velocity-only.ipynb)
- **TODO** make other notebooks ready for colab

### Code Layout
- `docs/`: LaTeX source of the thesis
- `src/`: Python/Φ<sub>Flow</sub> source code, imported in the notebooks
- `*.ipynb`: interactive notebook files
- `original-code/`: (see below)
- `taichi/`: (see below)

### Taichi Version
The `taichi` folder contains a minimal [taichi](https://docs.taichi-lang.org/)
implementation of visualizing the first 100 basis fields.

#### Install Taichi
```
pip install taichi==1.2.1
```

#### Run the Taichi Version
```
python taichi/main.py
```
- Viscosity
    - `Y`: turn ON
    - `T`: turn OFF 
- Step basis fields
    - `J`: Next basis field (increment)
    - `K`: Previous basis field (decrement)
- `V`: Visualize (plot) current velocity field with matplotlib
    - the title displays the current base coefficient vector (doesn't display
        properly if many basis fields are used)
- `R`: Random basis coefficients

Hop into `taichi/main.py`, to change the number of basis fields. 
(Look for `N=...`.)

### Original Code by de Witt et al 
The `original-code` folder contains the original implementation of *Fluid
Dynamics using Laplacian Eigenfunctions* by Tyler de Witt, Christian Lessig, and
Eugene Fiume, downloaded from the project's website:
[http://www.dgp.toronto.edu/~tyler/fluids/](http://www.dgp.toronto.edu/~tyler/fluids/).
In the early phases of the project, it was used to print values for checking the
correctness of our implementation.
```
cd original-code
javac LE.java #compile 
java LE #run 
java LE > output #print to an output file
python visu.py #plot the velocity field with matplotlib for comparison
```

(For a (more involved) C++ implementation, check out the [source
code](https://bitbucket.org/cqd123123/eigenfluidrelease/src/release/) for the
paper [Scalable Laplacian
Eigenfluids](https://w2.mat.ucsb.edu/qiaodong/SIG18-EigenFluid/index.html).)


