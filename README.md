# PEKET
``peket``: **P**yCBC **E**xtension for **K**ilonova **E**mission **T**argeting

![logo](logo.png)

**NOTE**: This repo is no more active, this is a **legacy** code - The final version of the search pipeline will be moved to GitLab (LIGO) when finalized

## Overview
[!WARNING] 
>"REMOVED CLI"
>All the CLI where removed from the `pyproject.toml` as they are no longer supported.
>    
>*(Note: The source code is still avaiable).*

PEKET is an add-on package for PyCBC and NMMA/FIESTA. It provides command-line tools to bridge Kilonova emission inferences with Gravitational Wave coherent searches:
* ``gw-setup-pipeline``: Performs coherent GW searches based on properties inferred from KNe.
* ``gw-search-significance``: Estimates the statistical significance (FAR and p-value) of the top GW trigger using time slides.
* ``kn-make-grid``: Generates grids of synthetic kilonovae lightcurves.
* ``kn-ts-loop``: Performs iterative inference to track timeshift evolution.

## Quick Install
```bash
git clone -b main git@github.com:NJamsin/PEKET.git
cd PEKET
pip install -e .
```

Read the full documentation [here](https://NJamsin.github.io/peket/)

### Acknowledgements
I would like to acknowledge the use of AI for this project.