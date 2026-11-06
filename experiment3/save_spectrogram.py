"""Save windowed GC/PDC/DTF for one LIF condition using connectivity_spectra."""

from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .population_connectivity import _stable, choose_order, connectivity_spectra, fit_var
else:
    from population_connectivity import _stable, choose_order, connectivity_spectra, fit_var


BASE = Path(__file__).resolve().parent
METRICS = ("GC", "PDC", "DTF")


@dataclass(frozen=True)
class Spectrogram:
    times: np.ndarray
    frequencies: np.ndarray
    measures: dict  # metric -> (window, direction [1->2, 2->1], frequency)
    errors: np.ndarray
    order: int
    window_s: float
    step_s: float


def compute_spectrogram(counts, fs, *, window_s=2., step_s=.16, p_max=20):
    """Standardize counts over the record, select BIC order, refit each window."""
    y = np.asarray(counts, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all():
        raise ValueError("Нужны два конечных ряда чисел спайков: (time, 2).")
    if any(not np.isfinite(value) or value <= 0 for value in (fs, window_s, step_s)):
        raise ValueError("Частота, длина окна и шаг должны быть положительными.")
    if not np.isfinite(p_max) or p_max < 1 or int(p_max) != p_max:
        raise ValueError("p_max должен быть положительным целым числом.")
    if np.any(y.std(axis=0) == 0):
        raise ValueError("Одна из популяций не спайкует: спектрограмма не оценивается.")
    window, step = round(window_s*fs), max(1, round(step_s*fs))
    if window > len(y):
        raise ValueError("Длина окна превышает длительность записи.")
    y = (y-y.mean(axis=0))/y.std(axis=0)
    order = choose_order(y, int(p_max), criterion="BIC")
    if window-order <= 2*order+1:
        raise ValueError(f"Окно слишком короткое для VAR({order}); увеличьте --window.")
    starts = np.arange(0, len(y)-window+1, step)
    frequencies = np.linspace(0, fs/2, 1001)
    measures = {name: np.full((len(starts), 2, len(frequencies)), np.nan) for name in METRICS}
    errors = []
    for index, start in enumerate(starts):
        try:
            weights, covariance = fit_var(y[start:start+window], order)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if not _stable(weights) or eigenvalues[0] <= 1e-10*eigenvalues[-1]:
                raise ValueError("Неустойчивая VAR или вырожденные инновации в окне.")
            spectra = connectivity_spectra(weights, covariance, frequencies, fs)
            if not all(np.isfinite(values).all() for values in spectra.values()):
                raise ValueError("Неконечные спектральные оценки в окне.")
            for name in METRICS:
                measures[name][index] = spectra[name]
            errors.append("")
        except (ValueError, np.linalg.LinAlgError) as error:
            errors.append(str(error))
    if not any(error == "" for error in errors):
        raise ValueError("Нет допустимых окон для спектрограммы.")
    return Spectrogram((starts+window/2)/fs, frequencies, measures, np.asarray(errors),
                       order, window/fs, step/fs)


def save_figure(result, config, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True,
                                constrained_layout=True)
    for column, metric in enumerate(METRICS):
        data = result.measures[metric]
        finite = data[np.isfinite(data)]
        maximum = max(float(finite.max()), np.finfo(float).eps)
        for direction, label in enumerate(("1 → 2", "2 → 1")):
            axis = axes[direction, column]
            # Explicit time edges also support records with only one full window.
            time_edges = np.r_[result.times-result.step_s/2, result.times[-1]+result.step_s/2]
            df = result.frequencies[1]-result.frequencies[0]
            frequency_edges = np.r_[result.frequencies-df/2, result.frequencies[-1]+df/2]
            mesh = axis.pcolormesh(time_edges, frequency_edges,
                                   np.ma.masked_invalid(data[:, direction].T),
                                   shading="flat", cmap="turbo", vmin=0, vmax=maximum)
            axis.axhline(config.frequency, color="black", linewidth=.6, alpha=.5)
            axis.set(title=f"{metric}: {label}", xlim=(time_edges[0], time_edges[-1]), ylim=(0, config.fs/2))
            if direction == 1:
                axis.set_xlabel("Время от начала симуляции, с")
            if column == 0:
                axis.set_ylabel("Частота, Гц")
        figure.colorbar(mesh, ax=axes[:, column], label=metric, shrink=.9)
    figure.suptitle(
        f"Гармонический ток: A = {config.rhythm_amplitude:g} пА, "
        f"задержка = {1000*config.coupling_delay:g} мс, "
        f"пуассоновский вход = {config.noise:g} Гц, seed = {config.seed}\n"
        f"Числа спайков; BIC: VAR({result.order}); окно {result.window_s:g} с, "
        f"шаг {result.step_s:g} с; общая нормировка по записи")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=200,
                        help="Peak amplitude of the harmonic current, pA.")
    parser.add_argument("--delay-ms", type=float, default=40)
    parser.add_argument("--noise", type=float, default=50, help="Rate of each external Poisson source, Hz.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--window", type=float, default=2)
    parser.add_argument("--step", type=float, default=0.16)
    parser.add_argument("--p-max", type=int, default=20)
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--output-dir", type=Path, default=BASE/"direction_results_harmonic")
    args = parser.parse_args(argv)

    if __package__:
        from .spike_population_brian2 import LIFConfig, _validate, simulate_lif_populations
    else:
        from spike_population_brian2 import LIFConfig, _validate, simulate_lif_populations
    config = LIFConfig(rhythm_amplitude=args.amplitude, coupling_delay=args.delay_ms/1000,
                       noise=args.noise, duration=args.duration, seed=args.seed)
    try:
        _validate(config)
    except ValueError as error:
        parser.error(str(error))
    if args.backend == "cython":
        import brian2 as b2
        b2.prefs.codegen.runtime.cython.cache_dir = "/tmp/brian2_v3_harmonic_spectrogram_cython"
    print(f"Симуляция: {args.amplitude:g} пА, {args.delay_ms:g} мс, "
          f"вход {args.noise:g} Гц, seed={args.seed}, запись {args.duration:g} с.", flush=True)
    counts = simulate_lif_populations(config, backend=args.backend)
    result = compute_spectrogram(counts, config.fs, window_s=args.window, step_s=args.step,
                                 p_max=args.p_max)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = (f"spectrogram_{args.amplitude:g}pA_delay_{args.delay_ms:g}ms_"
            f"noise_{args.noise:g}Hz_seed{args.seed}")
    metadata = dict(config=asdict(config), backend=args.backend, burn_in_s=0.0,
                    drive=dict(waveform="harmonic", amplitude_convention="peak", phase_rad=0.0),
                    signal="binned_spike_counts", normalization="record_zscore",
                    order_criterion="BIC", order=result.order, p_max=args.p_max,
                    window_s=result.window_s, step_s=result.step_s,
                    directions=["1->2", "2->1"], metrics=list(METRICS),
                    window_significance_test=False,
                    source_hashes={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                   for name in ("spike_population_brian2.py", "population_connectivity.py",
                                                "save_spectrogram.py")})
    np.savez_compressed(output/(stem+".npz"), metadata_json=json.dumps(metadata, ensure_ascii=False),
                        counts=counts, times=result.times, frequencies=result.frequencies,
                        errors=result.errors, **result.measures)
    path = output/(stem+".png")
    save_figure(result, config, path)
    print(f"VAR({result.order}), допустимых окон: {np.count_nonzero(result.errors == '')}"
          f"/{len(result.times)}. Сохранено: {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
