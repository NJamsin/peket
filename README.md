# PEKET
``peket``: **P**yCBC **E**xtension for **K**ilonova **E**mission **T**argeting

![logo](logo.png)

**NOTE**: ``peket`` is still under development. Feel free to contact me if you have any question

## Overview
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
I would like to acknowledge the use of [Gemini](https://gemini.google.com/app) for fixing code errors and assisting with the logo design.