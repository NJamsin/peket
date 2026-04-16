# gw-setup-pipeline

This command-line tool performs coherent GW searches (using `pycbc_multi_inspiral` as its backend) based on properties inferred from KNe.

## Usage
``gw-setup-pipeline`` requires one **OBLIGATORY** argument: the path to the ``.yaml`` config file.

```bash
gw-setup-pipeline path/to/config.yaml [OPTIONS]
```

## Configuration File Structure (.yaml)
The structure of the config file should be as follows:
```yaml
Directory:
  BASE_DIR: # Directory created for the pipeline. Ideally use one directory per run.
  run_name: # A unique name for this run, used for naming outputs and logs.

KN_data:
  first_detection: # ISOT format (yyyy-mm-ddThh:mm:ss.). Ensure H1/L1 were taking data!
  ra: # Right Ascension in radians
  dec: # Declination in radians
  EM_post_file: # Path to EM posterior samples (requires a timeshift column).
  RESAMP_post_file: # Path to RESAMP posterior samples (requires chirp_mass and mass_ratio).

GW_search:
  num_splits: # Number of splits for the template bank
  window_size: # Size of the max time window in seconds

Injection: # Only read if --injection is passed.
  mass1: # in solar mass
  mass2: # in solar mass
  distance: # Distance in Mpc
  ra: # Right Ascension (radians)
  dec: # Declination (radians)
  polarization: # Polarization angle
  approximant: # Waveform approximant (str)
  time_offset: # Seconds after the middle of the window for the merger.
```

## Optional Arguments
- ``--submit``: Automatically submit the pipeline to HTCondor after generation.
- ``--injection``: Injects a fake signal based on the 'Injection' section of the config.
- ``--expected-trigger-time``: Expected trigger time in gps format. Used in final plots.
- ``--skip-search``: Skips the search step and runs post-processing (requires existing triggers).
- ``--plot-spectrogram``: Generates a spectrogram plot for the top trigger.
  - ``--spectrogram-range``: vmin and vmax for the spectrogram plot. Only used if ``--plot-spectrogram`` is set, default values are ``vmin=0, vmax=15``.
- ``--detector-threshold``: Minimum antenna response to launch the search (default: 0.5). Only applied to injections.
- ``--plot-antenna-pattern``: Generates an antenna pattern plot for the source location (injections only).
- ``--template-bank``: Path to the template bank file if you want to specify it instead of generating through the resampling posterior.
- ``--monitor``: If true, will monitor the pipeline execution. (ignored if ``--skip-search`` is used).

### Known Issues
It is possible that an OOM (Out Of Memory) error kills the search jobs despite the 4GB of RAM requested for each job. Rerunning the command usually solves the problem.
