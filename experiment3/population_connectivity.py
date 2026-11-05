"""Bivariate VAR connectivity with a phase-surrogate significance test."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopulationConnectivity:
    frequencies: np.ndarray
    order: int
    measures: dict  # metric -> (direction [1->2, 2->1], frequency)
    thresholds: dict
    significant: dict


def fit_var(y, order, *, start=None, ridge=1e-6):
    """VAR with intercept, ML innovation covariance, and a small relative ridge."""
    start = order if start is None else start
    n, channels = y.shape
    response = y[start:]
    design = np.column_stack([
        np.ones(n - start),
        *[y[start - lag:n - lag] for lag in range(1, order + 1)],
    ])
    if len(response) <= design.shape[1]:
        raise ValueError("Слишком мало отсчётов для VAR.")
    gram = design.T @ design
    penalty = ridge * np.trace(gram[1:, 1:]) / (design.shape[1] - 1)
    regularizer = penalty * np.eye(design.shape[1])
    regularizer[0, 0] = 0  # Do not penalize the intercept.
    coefficients = np.linalg.solve(gram + regularizer, design.T @ response)
    residuals = response - design @ coefficients
    covariance = residuals.T @ residuals / len(response)
    weights = coefficients[1:].reshape(order, channels, channels).transpose(0, 2, 1)
    return weights, covariance


def _stable(weights):
    order, channels, _ = weights.shape
    companion = np.zeros((channels * order, channels * order))
    companion[:channels] = np.hstack(weights)
    if order > 1:
        companion[channels:, :-channels] = np.eye(channels * (order - 1))
    return np.abs(np.linalg.eigvals(companion)).max() < 1 - 1e-10


def choose_order(y, p_max=20, *, criterion="BIC"):
    """AIC/BIC candidates use the same response samples and an identical estimator."""
    criterion = criterion.upper()
    if criterion not in ("AIC", "BIC"):
        raise ValueError("Критерий порядка VAR должен быть AIC или BIC.")
    p_max = min(p_max, (len(y) - 2) // 3)
    if p_max < 1:
        raise ValueError("Недостаточно данных для выбора порядка VAR.")
    scores = np.full(p_max, np.inf)
    for order in range(1, p_max + 1):
        try:
            weights, covariance = fit_var(y, order, start=p_max)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign > 0 and np.isfinite(logdet) and _stable(weights):
                n = len(y) - p_max
                penalty = 2 if criterion == "AIC" else np.log(n)
                scores[order - 1] = logdet + penalty * (4 * order + 2) / n
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not np.isfinite(scores).any():
        raise ValueError("Нет устойчивой VAR с невырожденными инновациями.")
    return int(scores.argmin() + 1)


def connectivity_spectra(weights, covariance, frequencies, fs):
    """Geweke GC and amplitude PDC/DTF; rows are targets, columns are sources."""
    phase = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(1, len(weights) + 1) / fs)
    ar = np.eye(2)[None] - np.einsum("fp,pij->fij", phase, weights)
    transfer = np.linalg.inv(ar)
    power = transfer @ covariance @ transfer.conj().transpose(0, 2, 1)
    measures = {name: np.empty((2, len(frequencies))) for name in ("GC", "PDC", "DTF")}
    for direction, (source, target) in enumerate(((0, 1), (1, 0))):
        source_variance = (
            covariance[source, source] - covariance[source, target] ** 2 / covariance[target, target]
        )
        transferred = source_variance * np.abs(transfer[:, target, source]) ** 2
        intrinsic = 1 - transferred / power[:, target, target].real
        if np.any((intrinsic < -1e-8) | (intrinsic > 1 + 1e-8)):
            raise ValueError("Некорректное разложение мощности для GC.")
        measures["GC"][direction] = -np.log(np.clip(intrinsic, np.finfo(float).eps, 1))
        measures["PDC"][direction] = np.abs(ar[:, target, source]) / np.linalg.norm(ar[:, :, source], axis=1)
        measures["DTF"][direction] = np.abs(transfer[:, target, source]) / np.linalg.norm(transfer[:, target, :], axis=1)
    return measures


def analyze_population_connectivity(
    counts, fs, *, p_max=20, order_criterion="BIC", n_surrogates=199, alpha=0.05, seed=42,
):
    """Analyze binned spike counts; the sign of an imposed delay is never inspected."""
    y = np.asarray(counts, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all():
        raise ValueError("Нужны два конечных временных ряда: (time, 2).")
    if np.any(y.std(axis=0) == 0):
        raise ValueError("Одна из популяций не спайкует: направление оценить нельзя.")
    if not 0 < alpha < 1:
        raise ValueError("alpha должна быть между 0 и 1.")
    critical_rank = int(np.ceil((n_surrogates + 1) * (1 - alpha / 3)))
    if n_surrogates < 1 or critical_rank > n_surrogates:
        raise ValueError("Недостаточно суррогатов для заданного alpha.")
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    order = choose_order(y, p_max, criterion=order_criterion)
    frequencies = np.linspace(0, fs / 2, 1001)
    weights, covariance = fit_var(y, order)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
        raise ValueError("Независимые инновации каналов не определяются.")
    measures = connectivity_spectra(weights, covariance, frequencies, fs)

    rng = np.random.default_rng(seed)
    fourier = np.fft.rfft(y, axis=0)
    null_maxima = np.empty((n_surrogates, 3))
    for repetition in range(n_surrogates):
        phases = rng.uniform(0, 2 * np.pi, size=fourier.shape)
        randomized = np.abs(fourier) * np.exp(1j * phases)
        randomized[0] = fourier[0]
        if len(y) % 2 == 0:
            randomized[-1] = fourier[-1] * rng.choice([-1, 1], size=2)
        surrogate = np.fft.irfft(randomized, n=len(y), axis=0)
        sw, sc = fit_var(surrogate, order)
        spectra = connectivity_spectra(sw, sc, frequencies, fs)
        null_maxima[repetition] = [spectra[name].max() for name in measures]

    thresholds, significant = {}, {}
    for metric, name in enumerate(measures):
        maxima = null_maxima[:, metric]
        observed = measures[name].max(axis=1)
        thresholds[name] = float(np.sort(maxima)[critical_rank - 1])
        p_values = np.minimum(
            1, 3 * (1 + (maxima[:, None] >= observed).sum(axis=0)) / (n_surrogates + 1)
        )
        significant[name] = p_values <= alpha
    return PopulationConnectivity(
        frequencies=frequencies, order=order, measures=measures,
        thresholds=thresholds, significant=significant,
    )
