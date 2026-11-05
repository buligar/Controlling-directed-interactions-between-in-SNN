# Controlling directed interactions between spiking neural populations by changing external excitation

This repository studies directed interactions in LIF neural networks under changing
external excitation and fixed synaptic topology. Networks are simulated in Brian2
and analyzed using spectral Granger causality (GC), partial directed coherence (PDC),
and the directed transfer function (DTF). See the [paper](article.pdf)
by B. B. Batuev and S. V. Sukhov for details.

## Installation and usage

```bash
python3 -m pip install numpy scipy matplotlib brian2 networkx cython
```

Run commands from the repository root. Cython requires a C/C++ compiler;
use `--backend numpy` in Experiment 3 to run without compilation.

| Script | Experiment |
| --- | --- |
| [experiment1.py](experiment1.py) | Three populations of 50 neurons; stimulation switches at 10 → 20 → 30 Hz, PDC spectrograms |
| [experiment2.py](experiment2.py) | Two populations; separate sweeps of current amplitude and Poisson input rate, 20 realizations, bootstrap inference |
| [experiment3/systematic_analysis.py](experiment3/systematic_analysis.py) | Amplitude, noise, and delay sweep; 560 simulations, spike count analysis, phase surrogates |

```bash
python3 experiment1.py
python3 experiment2.py
python3 experiment3/systematic_analysis.py --output-dir experiment3/direction_results
```

The first script saves figures in the current directory; the second saves them
in `results_brian2_var_style/`. Set parameters for Experiments 1–2 in the code
and for Experiment 3 through command-line arguments (`--help`).

Saved results: [Experiment 1](experiment1_results/),
[Experiment 2](experiment2_results/),
[Experiment 3](experiment3/experiment3_results/).
Regenerate the six direction maps from `manifest.json` and `runs.csv` without Brian2:

```bash
python3 experiment3/systematic_analysis.py --report-only \
  --output-dir experiment3/experiment3_results
```

Per-run caches in `runs/` directories are excluded from Git.
Choose a new `--output-dir` when changing sweep settings or source code.

## Figures from the paper

All eight figures appear in the same order as in the paper, converted from the original PDFs to PNG.

**Figure 1.** Switching external excitation among three populations.

![Stimulation protocol for three populations](docs/figures/3population.png)

**Figure 2.** Network topology, neuron types, and external projections for two populations.

![Network topology and external Poisson sources](docs/figures/topology_seed_20261009.png)

**Figure 3.** Workflow for sweeping current amplitude and Poisson input rate.

![Experiment 2 workflow](docs/figures/experiment_scheme.png)

**Figure 4.** PDC spectrograms during stimulation switching among three populations.

![PDC spectrogram matrix](docs/figures/PDC_matrix_ru.png)

**Figure 5.** Signed peaks of GC, PDC, and DTF asymmetry across 20 realizations;
shading shows confidence bands, and stars indicate significance after Holm correction.

![Directional asymmetry under amplitude and noise sweeps](docs/figures/delta_signed_peak.png)

**Figure 6.** GC directionality over 0–50 Hz (left) and 8–12 Hz (right);
each category requires a strict majority of five realizations.

![GC direction maps](docs/figures/delay_direction_GC.png)

**Figure 7.** PDC directionality over the full frequency range and within the rhythm band.

![PDC direction maps](docs/figures/delay_direction_PDC.png)

**Figure 8.** DTF directionality. With the bivariate normalizations used here,
DTF and PDC coincide, so matching maps do not provide independent confirmation.

![DTF direction maps](docs/figures/delay_direction_DTF.png)

## Interpretation

As harmonic current increases, PDC/DTF asymmetry reverses sign while GC asymmetry
remains positive. In Experiment 3, sufficiently strong rhythmic input generally
produces a predominant direction from the stimulated population to the other
within the 8–12 Hz band.

These metrics describe statistical dependencies between signals. The highest noise
level in Experiment 2 saturates the discrete `PoissonGroup` generator;
the sign of the delay parameter in Experiment 3 also selects the stimulated population.

Code license: [GNU AGPL v3.0](LICENSE). The paper's licensing terms are stated in the [PDF](article.pdf).
