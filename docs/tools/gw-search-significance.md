# gw-search-significance

This command-line tool estimates the statistical significance of the top trigger identified during the coherent search. It computes the False Alarm Rate (FAR) and p-value by generating background data via time slides on off-source windows.

## Usage
``gw-search-significance`` requires the path to the same ``.yaml`` config file used for the initial search.

```bash
gw-search-significance path/to/config.yaml [OPTIONS]
```
## Arguments 
- `config` (Obligatory): Path to the configuration file.
- `--n-slides`: Number of time slides to generate for background estimation. Default is 300.
- `--run-background`: Generates the HTCondor sub files and automatically submits the jobs for the background search.
- `--submit`: Auto-submits the background jobs to HTCondor (similar utility to ``--run-background``).
- `--monitor`: Monitors the execution of the background jobs with a live progress bar.
- `--window`: Specifies which off-source window(s) to use for background estimation. Choices are ``both``, ``before``, or ``after``. Default is ``both``.

## Output & Visualizations
Upon successful execution, the tool will evaluate the background triggers against the top candidate and output:
1. Statistical metrics: Top trigger ranking statistic, total background time analyzed, FAR, and the on-source p-value.
2. Timeline Plot: Saved as resamp_bank_timeline.png in your plots directory, visualizing the on-source and off-source windows.
3. FAR vs SNR Plot: Saved as resamp_bank_far_vs_snr.png to visually assess the significance of the event against the background distribution.

## Example
Here is an example of running a background estimation using 3 time slides on both off-source windows, automatically submitting and monitoring the HTCondor jobs:
```bash
gw-search-significance /path/to/search_config.yaml --n-slides 3 --submit --monitor --run-background
```
**Console Output:**
```
On-source window (GPS): 1187006504 - 1187008913
Off-source windows (GPS):
  Window 1: 1187008175 - 1187006488
  Window 2: 1187008929 - 1187011338

Background data files already exist for both detectors. Skipping download and preparation.

=======================================================
  PEKET - Significance estimation (Time Slides)
=======================================================

Top trigger ranking stat : 22.2409 at epoch 1187008882.4 (GPS)

Generating 3 time slides across 2 off-source windows (size 2409s each)...
Generated time slides file: resamp_bank_sig_windows_1.txt with 3 slides × 10 banks in chunks of 1000s with 16s overlap. Total jobs: 90
Generated time slides file: resamp_bank_sig_windows_2.txt with 3 slides × 10 banks in chunks of 1000s with 16s overlap. Total jobs: 90
  Merged time slides file created: resamp_bank_sig_windows_all.txt

Generating timeline plot...
  Timeline plot saved to: /gw_search/plots/resamp_bank_timeline.png
  HTCondor files generated: run_significance_search.sh
                            significance_search_both.sub
Submitting job(s)....................................................
180 job(s) submitted to cluster 552.
--------------------------------------------------
Monitoring background jobs (This will take time due to coherent search)...
--------------------------------------------------
[████████████████████] 180/180 background jobs done (100%)
  All background jobs completed.

Collecting background triggers from /gw_search/significance/out...

──────────────────────────────────────────────────
  Top trigger stat     : 22.2409
  Louder than top      : 0
  T_background         : 7179.0 s  (0.000 yr)
  FAR                  : < 1.393e-04 Hz  (< 4396.155 /yr)
  p-value (on-source)  : < 2.851e-01
──────────────────────────────────────────────────

Generating FAR vs SNR plot...
  FAR vs SNR plot saved to: /gw_search/plots/resamp_bank_far_vs_snr.png
Significance estimation complete.
```