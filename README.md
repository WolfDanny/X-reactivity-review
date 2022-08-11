[![DOI](https://zenodo.org/badge/369218091.svg)](https://zenodo.org/badge/latestdoi/369218091) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Generating recognition networks and simulating heterologous infection

This repository contains:
* Python codes used to:
    * generate bipartite recognition networks and simulate a heterologous infection (`Recognition-network-gillespie.py`),
    * plot the results of the simulation (`Plots.py`).
* Seed for random number generation used to produce the published figures (`seed.bin`).
* Environment file to create the Anaconda environment used (`environment.yml`).

## Creating the Anaconda environment

To create and activate the Anaconda environment used run the following commands from the current directory:
```bash
conda env create -f environment.yml
conda activate cross-reactivity
```

## Using the python codes

* To reproduce the published figures run `Recognition-network-gillespie.py` followed by `Plots.py` for each of the values of the `network` parameter (`0, 1, 2`).
* To run a new simulation set the parameter `reproduce_last_result` to `False` (note that this will overwrite the current `seed.bin` file).
* To change the probability of peptide recognition change the parameter `peptide_probability` from `None` to the desired value.
