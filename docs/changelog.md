# Changelog

## Latest Updates 0.1.1

### Documentation Overhaul
* Completely restructured the documentation using **MkDocs** and the **Material for MkDocs** theme. 
* Separated CLI references, installation guides, and tutorials into dedicated, easily navigable pages.

### gw-search-significance (New CLI)
* Added a new command-line tool to estimate the False Alarm Rate (FAR) and p-value for the top trigger.
* Automatically generates time slides across off-source windows to compute the background distribution.
* Features automatic HTCondor job submission, live progress monitoring, and automatic generation of timeline and FAR vs SNR plots.

### kn_side.utils (LSST Integration)
* Introduced comprehensive utilities to generate synthetic kilonova lightcurves simulating Vera C. Rubin Observatory (LSST) observations.
* Integrated `rubin_sim` to fetch realistic cadences, filters, and 5-sigma depths from LSST databases.
* Added functions to ensure simulated sources fall within the realistic Field of View (FOV) and explode during expected lifetimes.
* Automatically calculates observed magnitudes with realistic Gaussian noise and upper limits handling.

### Others
Minor correction to other CLIs (cleaning of unused lines).
---

## Version 0.1.0
Found a name and reorganized the structure. Beta version available on the `package` branch of the old [repo](https://github.com/NJamsin/KN_GW-BNS-master/tree/package).

### gw-setup-pipeline
* Completely changed the pre-treatment of the `.gwf` files. Now uses cache files (`.lcf`) -> reduces the computing time.
* Modified the `--monitor` argument to correctly print errors.
* Added a `--detector-threshold` argument for injection to early-stop the search if the detector response is too low with respect to the sky position.

### kn-make-grid
* Added supports for `Bu2019lm` and `Ka2017`.
* Corrected a small mistake that could lead to non-physical parameter values.

### kn-ts-loop
* Added a new command-line tool to iteratively perform inference by progressively removing early detection points.
* Supports optional upper limits substitutions and automatic GW resampling and corner plotting.
* Enhanced robustness of the timeshift loop to avoid skipping resampling in case of early-stop.