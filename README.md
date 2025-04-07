# AsymptoticRingModeSolver
Code package to calculate the mode distributions for an integrated micro-resonator side coupled to a bus waveguide using the asymptotic field formalism. Provides a self-contained solution for generating mode properties utilizing the open source Femwell mode solver, and applying the resulting properties in computing the linear dynamics of the asymptotic field modes.

## Installation
This package requires the open source Fewmell package (found here: https://github.com/HelgeGehring/femwell/tree/main). Once installed, clone the AsymptoticRingModeSolver with:
```
git clone git@github.com:mic-sloan/AsymptoticRingModeSolver.git
```
Then pip install the package as:
```
pip install .
```

## Examples
Examples on how to generate the mode properties needed for the asymptotic mode solver, as well as examples of the asymptotic field computations can be found in the examples directory.
