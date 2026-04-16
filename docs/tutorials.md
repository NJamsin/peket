# Tutorials & Utilities

This section provides information on the tutorials available in the repository, as well as a detailed reference for the Python utility functions included in the `peket.kn_side.utils` module.

---

## Available Tutorials
Examples and tutorials are provided in the GitHub repository under the [`example_file`](https://github.com/NJamsin/PEKET/tree/main/example_file) directory:

* **Real KN-GW Data:** Configuration for analyzing AT2017gfo-GW170817. Outputs are located in the [`test_real_data`](https://github.com/NJamsin/PEKET/tree/main/test_real_data) directory.
* **Artificial Injection:** A KN generated with NMMA (`Bu2019lm`) with an injected GW in real H1/L1 strain data. Outputs are located in [`test_injection`](https://github.com/NJamsin/PEKET/tree/main/test_injection).
* **Timeshift Loop Tutorial:** A guide on combining `kn-ts-loop`, `kn-make-grid`, and `plot_param_evolution` is available in the [`example_file/ts-loop`](https://github.com/NJamsin/PEKET/tree/main/example_file/ts-loop) directory. It includes example prior files for all supported models.

---

## Utilities (`peket.kn_side.utils`)

The package provides several Python functions designed to help you generate synthetic data and plot your results.

### Plotting Utilities

#### `plot_param_evolution`
Generates summary plots for the complete analysis of a grid processed with `kn-ts-loop`. It visualizes how the inferred timeshift and other EM parameters evolve as early data points are removed.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `MODEL` | `str` | *Obligatory* | Name of the EM model (e.g., `'Bu2026_MLP'`). |
| `DIR` | `str` | *Obligatory* | Directory where the timeshift loop results are stored. |
| `UL` | `bool` | `False` | Set to `True` if the loop was run with upper limits substitution. |
| `true_merger` | `str` | `'2020-01-07T00:00:00.000'` | ISOT time of the true merger. |
| `minus_num` | `int` | `4` | Number of early points progressively removed during the loop. |
| `ts_max` | `float` | `-2.5` | Maximum timeshift value (negative) to be plotted. |
| `col_num` | `int` | `5` | Number of columns in the plot grid. |
| `row_num` | `int` | `5` | Number of rows in the plot grid. |

*Returns:* `None`. Automatically saves plots (evolution and injection-recovery P-P plots) in `{DIR}/plots/`.

---

### LSST Synthetic Lightcurve Generation
These functions interact with `rubin_sim` to simulate realistic Vera C. Rubin Observatory (LSST) cadences and limiting magnitudes.

#### `get_lsst_observations`
Retrieves and sorts LSST observations from a SQLite database.
* **Parameters:** `db_path` (str), `full_df` (bool, default `False`), `n_visits` (int, default `20`).
* **Returns:** A pandas DataFrame containing `expMJD`, `filter`, `fiveSigmaDepth`, `_ra`, and `_dec`.

#### `find_valid_sources`
Finds valid random sources in the LSST database that have observations during their expected kilonova lifetime and are located within the LSST field of view.
* **Parameters:** `df_lsst` (DataFrame), `n_sources` (int, default `5`), `duration` (int, default `20`), `min_observations` (int, default `3`).
* **Returns:** A list of dictionaries, each containing the source's `ra`, `dec`, `t0_mjd`, and its relevant `observations`.

#### `generate_synth_lc_lsst`
Generates a full synthetic lightcurve based on LSST observations of a given source and EM model, applying realistic 5-sigma depth noise and upper limits.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `dict` | *Obligatory* | Dictionary containing LSST observations (output from `find_valid_sources`). |
| `model_name` | `str` | *Obligatory* | Name of the EM model (e.g., `'Bu2026_MLP'`, `'Bu2019lm'`). |
| `model_param` | `dict` | *Obligatory* | Physical parameters for the chosen model. |
| `save` | `bool` | `False` | Whether to save the generated lightcurve to a file. |
| `filename` | `str` | `'test_lc_lsst.dat'` | Output filename if `save=True`. |
| `svd_path` | `str` | *Custom* | Path to SVD models (used if model is not `Bu2026_MLP`). |

*Returns:* A pandas DataFrame representing the fully formatted synthetic lightcurve ready for NMMA.

#### Internal Helpers
* **`get_source_observations`**: Extracts specific LSST observations relevant to a source's spatial and temporal position.
* **`get_obs_mag_from_lsst`**: Computes the observed magnitude based on the 5-sigma depth using `rubin_sim.phot_utils`.
* **`build_dic`**: Utility to structure observation arrays and depths per filter.