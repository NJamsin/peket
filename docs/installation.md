# Installation Guide

## 1. Prerequisites
Before installing PEKET, ensure you have the following dependencies correctly set up in your environment:

!!! warning "HTCondor Requirement"
    ``gw-setup-pipeline`` heavily relies on [HTCondor](https://htcondor.readthedocs.io/en/latest/) to create multiples `.sub` or `.dag` files. Make sure HTCondor is installed on your cluster/device. 
    
    Dynamic slots are recommended as the main job reserves **16 Gb of RAM**, and sub-jobs reserve **4 Gb**.
    *(Note: HTCondor is not required if you only plan to use `kn-make-grid` and `kn-ts-loop`).*

!!! warning "Important: Python Version"
    Build [FIESTA](https://github.com/nuclear-multimessenger-astronomy/fiestaEM/tree/main) and [NMMA](https://github.com/nuclear-multimessenger-astronomy/nmma) from source in the same environment. 
    **NMMA explicitly requires Python 3.12.**

!!! tip "LSST Utilities (Optional)"
    If you plan to use the LSST synthetic lightcurve generation utilities (`generate_synth_lc_lsst`), you will need the Rubin Observatory simulation framework. It is highly recommended to install it via `conda-forge` to gracefully handle its complex dependencies:
    ```bash
    conda install -c conda-forge rubin_sim
    ```
    You also need to run
    ```bash
    rs_download_data -d throughputs
    ```
    To get the throughputs files of the filters needed for the LSST related functions.
---

## 2. Source Code Modifications
!!! info "Ensure compatibility"
    To ensure full compatibility with PEKET, minor modifications to NMMA and PyMultiNest are required. These modifications are minimal and will not alter the standard functioning of either package.

**NMMA Modification 1 (`nmma/em/model.py`, line 637):**
Allows generating Bu2019lm lightcurves with inclinationEM instead of KNtheta.
```python
def generate_lightcurve(self, sample_times, parameters, filters = 'all'):
        parameters = self.parameter_conversion(parameters) # <-- THIS ONE 
        parameters_list = self.em_parameter_setup(parameters)
```

**NMMA Modification 2 (`nmma/post_processing/plotting_routines.py`, line 160):**
Explicitly names the argument for corner plots.
```python
corner_plot(plot_samples.T, labels, limits, outdir=outdir) # <-- Explicitly name the argument
```

**PyMultiNest Modification (`pymultinest/analyse.py`, line 34):**
Prevents Fortran reading issues.
```python
def loadtxt2d(intext): 
    try:
        return numpy.loadtxt(intext, ndmin=2)
    except ValueError:
        # Catch Fortran formatting missing the 'e'
        with open(intext, 'r') as f:
            text = f.read()
        text = re.sub(r'(?<=\d)(-[1-3]\d\d)', r'e\1', text) 
        text = re.sub(r'(?<=\d)(\+[1-3]\d\d)', r'e\1', text)
        try:
            return numpy.loadtxt(StringIO(text), ndmin=2)
        except:
            return numpy.loadtxt(StringIO(text))
    except:
        return numpy.loadtxt(intext)
```

## 3. Install PEKET
Once prerequisites and modifications are done, clone and install the package:
```bash
git clone -b main git@github.com:NJamsin/PEKET.git
cd PEKET
pip install -e .
```