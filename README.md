# Numerical simulation of the evolution of the pacreatic cancer and the axons regulation.

## Python modules required
---
Numpy (probably already installed):
```sh
$ pip install --user numpy
```
---
Numba:
```sh
$ pip install --user numba
```
---
Matplotlib:
```sh
$ pip install --user matplotlib
```
## Main features

The main features of the library are:
* the numerical computation of the coupled dynamical system modeling the evolution of the pancreatic cancer and axons' remodeling,
*  the numerical computation of in silico denervation on the system,
* the visualization of the evolution of the solution.
More details about the dynamical system can be found in [CCHMMP](https://hal.archives-ouvertes.fr/hal-02263522).

### File functions_pde_axons.py

Library of functions in order to compute the examples. The functions are organized in the following categories:
* the setting of the parameters,
* the discritization and computation of the coupled PDE and ODE equations,
* the in silico denervation,
* the visualization of the results.

### File example_pde_axons.py

The file "example_pde_axons.py" is a python program which computes the dynamical system, computes the system in silico denervated at defined times and plots the results: control solution and denervated solution and the axons (see [CCHMMP](https://hal.archives-ouvertes.fr/hal-02263522) for more details).
Three sets of parameters are given, one can choose between these sets and modify the times of denervation. The program returns three figures :
* the evolution of the axons,
* the evolution of the total density of cancer cells,
* the evolution of the distribution of the cells over the phenotype axis.




## Run on local machine 
```sh
$ python example_pde_axons.py
```

## Documentation links

* [numba](https://numba.pydata.org/) - Python library to accelerate the runtime (Python)
* [CCHMMP](https://hal.archives-ouvertes.fr/hal-02263522) - A continuous approach of modeling tumorigenesis and
axons regulation for the pancreatic cancer (Preprint)
