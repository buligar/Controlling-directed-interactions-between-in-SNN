"""
Амплитуда меняется в кластере 1, интенсивность пуассоновского шума — в кластере 2.
Для каждого seed все уровни используют одну топологию и одни случайные числа.
PLOT_TOPOLOGY/PLOT_SPECTRA/PLOT_SPIKES включают графики; SHOW_PLOTS — окна.
"""

# True — показывать окна Matplotlib, False — только сохранять PNG.
# Этот флаг задаётся до импорта pyplot и Brian2, чтобы выбрать backend.
SHOW_PLOTS = False

import matplotlib
if not SHOW_PLOTS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from brian2 import *
import numpy as np
import numpy.linalg as npl
from collections import Counter
from pathlib import Path

# Параметры моделирования: единицы мВ, пА, секунды и Гц
INFLUENCE_SCALES = (1, 2, 3, 4, 5, 6, 8, 10)
seeds = 20
seed_start = 20261009
t_total_sim = 10.0
dt_sim = 0.01
sampling_frequency = 100.0
discard = 1.0
n_cluster = 50
BASE_AMPLITUDE = 100.0
SWEEP_AMPLITUDE = 100.0
BASE_NOISE = 1
SWEEP_NOISE = 10
driving_frequency = 10.0
tau = 0.02
resistance = 0.08
rest = -65.0
threshold = -50.0
reset = -70.0
jump = 3.0
p_intra = 0.15
p_cross = 0.05
p_external = 0.3
excitatory_fraction = 0.8
win_len = 50
step = 16
p_max = 10
order = 0
ridge = 0.0  # Выбор порядка: OLS без регуляризации, как в var.py
var_ridge = 1e-3  # Оценка спектров: регуляризация из var_ols_const в var.py
nfft = 128
fmin = 0.0
fmax = 50.0
resamples = 9999
alpha = 0.05
min_valid = 0.6  # Минимальная доля допустимых окон, общих для всех амплитуд/уровней шума

# Число нейронов в двух кластерах
n_neurons = 2 * n_cluster

# Каталог результатов и настройки дополнительных графиков
out = Path('results_brian2_var_style')
PLOT_TOPOLOGY = True
PLOT_SPECTRA = True
PLOT_SPIKES = True
PLOT_SEED_INDEX = 0  # 0 — первая реализация; None — все seeds
PLOT_FACTORS = (1, 2, 3, 4, 5, 6, 8, 10)  # Выбранные influence_scale; None — все уровни
# Общий размер шрифта для всех рисунков, включая легенды и цветовые шкалы.
PLOT_FONT_SIZE = 14
plt.rcParams.update({
    'font.size': PLOT_FONT_SIZE,
    'axes.labelsize': PLOT_FONT_SIZE + 2,
    'axes.titlesize': PLOT_FONT_SIZE + 2,
    'xtick.labelsize': PLOT_FONT_SIZE,
    'ytick.labelsize': PLOT_FONT_SIZE,
    'legend.fontsize': PLOT_FONT_SIZE,
    'figure.titlesize': PLOT_FONT_SIZE + 4,
})
METRICS = ('GC', 'PDC', 'DTF')
SWEEPS = ('amplitude', 'noise')
prefs.codegen.target = 'numpy'

# Связи внутри кластеров и между кластерами, возбуждающие и тормозные нейроны
def make_topology(seed):
    """Undirected SBM -> reciprocal, signed E/I synapses, as in the originals."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    n = 2 * n_cluster
    block = np.arange(n) // n_cluster
    probability = np.where(block[:, None] == block[None, :], p_intra, p_cross)
    upper = np.triu(rng.random((n, n)) < probability, 1)
    adjacent = upper | upper.T
    is_exc = np.arange(n) % n_cluster < round(excitatory_fraction * n_cluster)
    signed = adjacent.astype(float) * np.where(is_exc[:, None], jump, -jump)
    external = rng.random((n, n_cluster)) < p_external
    ext_full = np.zeros((n, n), float)
    ext_full[:n_cluster, :n_cluster] = external[:n_cluster] * jump
    ext_full[n_cluster:, n_cluster:] = external[n_cluster:] * jump
    return (signed, ext_full)

# Моделирование потенциалов в Brian2
def lif_simulate(simulation_seed, signed, external, amplitudes, rates, record_spikes=False):
    """Brian2 Euler simulation with external PoissonGroup input.

    Reset Brian2's RNG to the same noise seed for every sweep level.
    PoissonGroup uses rand() < rates * dt, so dt must be small relative
    to the inverse of the highest input rate.
    With record_spikes=True, return (voltage trace, spike times in seconds,
    neuron indices); otherwise return only the voltage trace.
    """
    start_scope()
    prefs.codegen.target = 'numpy'
    noise_seed = int(np.random.SeedSequence([simulation_seed, 2]).generate_state(1)[0])
    seed(noise_seed)
    clock = Clock(dt=dt_sim * second)
    n = 2 * n_cluster
    namespace = dict(v_rest=rest * mV, 
                    v_threshold=threshold * mV, 
                    v_reset=reset * mV, 
                    resistance=resistance * mV / pA, 
                    tau=tau * second, 
                    frequency=driving_frequency * Hz)
    neurons = NeuronGroup(n,
                        '''dv/dt = (v_rest-v+resistance*amplitude*sin(2*pi*frequency*t))/tau : volt
                           amplitude : amp''',
                        threshold='v > v_threshold', 
                        reset='v = v_reset', 
                        method='euler', 
                        clock=clock, 
                        namespace=namespace)
    neurons.v = rest * mV
    neurons.amplitude = amplitudes * pA
    inputs = PoissonGroup(n, rates=rates * Hz, clock=clock)
    external_synapses = Synapses(inputs, neurons, 
                                 'weight : volt', 
                                 on_pre='v_post += weight',
                                 clock=clock)
    pre, post = np.nonzero(external)
    external_synapses.connect(i=pre, j=post)
    external_synapses.weight = external[pre, post] * mV
    recurrent_synapses = Synapses(neurons, neurons, 
                                  'weight : volt', 
                                  on_pre='v_post += weight', 
                                  clock=clock)
    pre, post = np.nonzero(signed)
    recurrent_synapses.connect(i=pre, j=post)
    recurrent_synapses.weight = signed[pre, post] * mV
    external_synapses.pre.order = -2
    recurrent_synapses.pre.order = -1
    monitor = StateMonitor(neurons, 
                           'v', 
                           record=True, 
                           dt=1 / sampling_frequency * second, 
                           when='start')
    network = Network(neurons, inputs, external_synapses, recurrent_synapses, monitor)
    if record_spikes:
        spike_monitor = SpikeMonitor(neurons)
        network.add(spike_monitor)
    network.run(t_total_sim * second, namespace={})
    voltage = np.asarray(monitor.v / mV)
    trace = np.column_stack((voltage[:n_cluster].mean(axis=0), voltage[n_cluster:].mean(axis=0)))
    if record_spikes:
        return (trace, np.asarray(spike_monitor.t / second),
                np.asarray(spike_monitor.i[:], dtype=int))
    return trace

# Амплитуды и шум для двух видов эксперимента
def input_settings(sweep, factor):
    if sweep == 'amplitude':
        amp = (SWEEP_AMPLITUDE * factor, SWEEP_AMPLITUDE)
        rate = (BASE_NOISE, BASE_NOISE)
    else:
        amp = (BASE_AMPLITUDE, BASE_AMPLITUDE)
        rate = (SWEEP_NOISE, SWEEP_NOISE * factor)
    return (np.repeat(amp, n_cluster), np.repeat(rate, n_cluster))

# Основные расчётные функции из var.py
def fit_var_ols_const(y, p, ridge=0.0):
    """OLS с константой; ковариация для информационных критериев делится на N."""
    y = np.asarray(y, float)
    T, k = y.shape
    N = T - p
    if p < 1 or N <= k * p + 1:
        raise ValueError('Слишком короткое окно для заданного p.')
    Y = y[p:]
    lags = [y[p-l-1:T-l-1] for l in range(p)]
    X = np.hstack([np.ones((N, 1)), *lags])
    XtX, XtY = X.T @ X, X.T @ Y
    B = (npl.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
         if ridge > 0 else npl.solve(XtX, XtY))
    c = B[0, :]
    A = np.stack([B[1+i*k:1+(i+1)*k, :].T for i in range(p)], axis=0)
    E = Y - X @ B
    Sigma = (E.T @ E) / N
    return A, c, Sigma


def companion_matrix(A):
    p, k, _ = A.shape
    top = np.hstack(A)
    if p == 1:
        return top
    I = np.eye(k * (p - 1))
    Z = np.zeros((k * (p - 1), k))
    return np.vstack([top, np.hstack([I, Z])])


def is_stable(A, tol=1.0-1e-10):
    return np.max(np.abs(npl.eigvals(companion_matrix(A)))) < tol


def information_criteria(y, p_max, ridge=0.0):
    y = np.asarray(y, float); T, k = y.shape
    AIC = []; BIC = []; HQ = []
    for p in range(1, p_max+1):
        try:
            A, c, Sigma = fit_var_ols_const(y, p, ridge=ridge)
            if not is_stable(A): raise RuntimeError
            ld = np.log(npl.det(Sigma)).real
            AIC.append(ld + (2.0/T)*p*k*k)
            BIC.append(ld + (np.log(T)/T)*p*k*k)
            HQ .append(ld + (2.0*np.log(np.log(T)))/T*p*k*k)
        except Exception:
            AIC.append(np.inf); BIC.append(np.inf); HQ.append(np.inf)
    return np.array(AIC), np.array(BIC), np.array(HQ)


def choose_var_order(v, win_len, step, p_max=20, criterion='BIC',
                     ridge=0.0, aggregator='mode', verbose=True):
    pstars = []
    for start in range(0, v.shape[0] - win_len + 1, step):
        seg = v[start:start + win_len]
        AIC, BIC, HQ = information_criteria(seg, p_max, ridge=ridge)
        ic = {'AIC': AIC, 'BIC': BIC, 'HQ': HQ}[criterion.upper()]
        if np.isfinite(ic).any():
            pstars.append(1 + np.argmin(ic))
    pstars = np.array(pstars, int)
    if pstars.size == 0:
        raise RuntimeError('Нет валидных окон для выбора порядка.')
    if aggregator == 'median':
        p_global = int(np.round(np.median(pstars)))
    elif aggregator == 'mean':
        p_global = int(np.round(np.mean(pstars)))
    else:
        counts = np.bincount(pstars, minlength=p_max + 1)
        counts[0] = 0
        p_global = int(np.flatnonzero(counts == counts.max()).min())
    if verbose:
        print(f'Рекомендуемый порядок VAR по {criterion}: p = {p_global}')
    return p_global, pstars


def spectra_2x2(W, Sigma, i, j, freqs, fs):
    """GC, DTF, PDC в направлении j → i по формулам var.py.

    Частотная сетка и fs передаются явно вместо глобальных переменных.
    Ортогонализация GC (T @ Hz) сохранена из исходного файла.
    """
    gc = np.empty(len(freqs))
    dtf = np.empty(len(freqs))
    pdc = np.empty(len(freqs))
    # s11, s22 = Sigma[0, 0], Sigma[1, 1]
    # s12, s21 = Sigma[0, 1], Sigma[1, 0]

    for l, f in enumerate(freqs):
        z = np.exp(-1j * 2 * np.pi * f / fs)
        Az = np.eye(2, dtype=complex)
        for p, Wl in enumerate(W, start=1):
            Az -= Wl * z**p
        Hz = npl.inv(Az)
        Sz = Hz @ Sigma @ Hz.conj().T
        Sii = np.real(Sz[i, i])
        source_variance = (
            Sigma[j, j] - Sigma[i, j]**2 / Sigma[i, i]
        )
        source_power = source_variance * np.abs(Hz[i, j])**2
        gc[l] = np.log(Sii / (Sii - source_power))
        dtf[l] = np.abs(Hz[i, j]) / np.sqrt(np.abs(Hz[i, 0])**2 + np.abs(Hz[i, 1])**2)
        pdc[l] = np.abs(Az[i, j]) / np.sqrt(np.abs(Az[0, j])**2 + np.abs(Az[1, j])**2)
    return gc, dtf, pdc


def var_ols_const(y, p=3, ridge=1e-3):
    """Оценка спектров как в var.py: фиксированный ridge, ковариация / dof."""
    y = np.asarray(y, float)
    T, k = y.shape
    dof = (T - p) - (k * p + 1)
    if p < 1 or dof <= 0:
        raise ValueError('Слишком короткое окно для заданного p.')
    Y = y[p:]
    lags = [y[p-l-1:T-l-1] for l in range(p)]
    X = np.hstack([np.ones((T - p, 1)), *lags])
    XtX, XtY = X.T @ X, X.T @ Y
    B = npl.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    W = np.stack([B[1+l*k:1+(l+1)*k].T for l in range(p)], axis=0)
    E = Y - X @ B
    Sigma = (E.T @ E) / dof
    return W, Sigma


def slide_2x2(v, i, j, order, freqs, fs, win_len=50, step=16, ridge=1e-3):
    """Расчёт из var.py; недопустимые окна сохраняют свои позиции как NaN."""
    starts = range(0, len(v) - win_len + 1, step)
    G, D, P = [np.full((len(freqs), len(starts)), np.nan) for _ in range(3)]
    times = np.array([(start + win_len // 2) / fs for start in starts])
    for w, start in enumerate(starts):
        try:
            W, Sigma = var_ols_const(v[start:start + win_len], order, ridge=ridge)
            with np.errstate(divide='raise', invalid='raise'):
                g, d, p = spectra_2x2(W, Sigma, i, j, freqs, fs)
            if np.all(np.isfinite([g, d, p])):
                G[:, w], D[:, w], P[:, w] = g, d, p
        except (npl.LinAlgError, ValueError, FloatingPointError):
            continue
    return G, D, P, times

# Расчёт спектров в скользящих окнах
def analyze_trace(trace, order, freqs):
    y = trace[round(discard * sampling_frequency):]
    gc_12, dtf_12, pdc_12, _ = slide_2x2(
        y, 1, 0, order, freqs, sampling_frequency, win_len, step, var_ridge)
    gc_21, dtf_21, pdc_21, _ = slide_2x2(
        y, 0, 1, order, freqs, sampling_frequency, win_len, step, var_ridge)
    maps = np.stack([np.stack([gc_12, gc_21], axis=1),
                     np.stack([pdc_12, pdc_21], axis=1),
                     np.stack([dtf_12, dtf_21], axis=1)], axis=1).transpose(3, 1, 2, 0)
    valid = np.all(np.isfinite(maps), axis=(1, 2, 3))
    maps[~valid] = np.nan
    return maps


def save_plot(fig, path):
    """Сохранить PNG; при SHOW_PLOTS оставить фигуру для показа в конце запуска."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches='tight')
    if not SHOW_PLOTS:
        plt.close(fig)
    print(f'График сохранён: {path}', flush=True)
    return path


def plot_topology(signed, external, simulation_seed):
    """Граф взаимных связей, знаковые веса E/I и внешние проекции."""
    n = signed.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_cluster, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    positions = np.vstack((ring + [-1.6, 0], ring + [1.6, 0]))
    is_exc = np.arange(n) % n_cluster < round(excitatory_fraction * n_cluster)
    pre, post = np.nonzero(np.triu((signed != 0) | (signed.T != 0), 1))
    segments = positions[np.column_stack((pre, post))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    ax = axes[0]
    ax.add_collection(LineCollection(segments, colors='0.45', linewidths=.5, alpha=.25))
    for center, label in ((-1.6, 'Population 1'), (1.6, 'Population 2')):
        ax.add_patch(Circle((center, 0), 1.1, fill=False, edgecolor='0.5', linestyle='--'))
        ax.text(center, 1.25, label, ha='center')
    for take, color, label in ((is_exc, '#1565c0', 'E: excitatory'),
                               (~is_exc, '#c62828', 'I: inhibitory')):
        ax.scatter(*positions[take].T, s=28, color=color, label=label, zorder=3)
    ax.set(xlim=(-2.9, 2.9), ylim=(-1.5, 1.5), aspect='equal')
    ax.set_title('Graph of network\n')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.2), frameon=False)
    ax.axis('off')

    limit = float(np.max(np.abs(signed))) or 1.0
    im = axes[1].imshow(signed, cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_title(f'Topology:' f'p_intra = {p_intra:g}, p_inter = {p_cross:g}')
    axes[1].set_xlabel('Postsynaptic neuron')
    axes[1].set_ylabel('Presynaptic neuron')
    fig.colorbar(im, ax=axes[1], shrink=.8, label='Synaptic weight, mV')
    im = axes[2].imshow(external, cmap='Blues',
                        vmin=0, vmax=float(np.max(external)) or 1.0)
    axes[2].set_title(f'External spike input p_input = {p_external:g}')
    axes[2].set_xlabel('Neuron in network')
    axes[2].set_ylabel('Source of external spikes')
    fig.colorbar(im, ax=axes[2], shrink=.8, label='Synaptic weight, mV')
    for ax in axes[1:]:
        ax.axhline(n_cluster - .5, color='0.25', linestyle='--', lw=1)
        ax.axvline(n_cluster - .5, color='0.25', linestyle='--', lw=1)
    return save_plot(fig, out / f'topology_seed_{simulation_seed}.pdf')


def plot_spikes(spike_times, spike_indices, simulation_seed, sweep, factor):
    """Растр спайков нейронов обоих кластеров за всё время моделирования."""
    spike_times = np.asarray(spike_times, float)
    spike_indices = np.asarray(spike_indices, int)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for cluster, color in enumerate(('#1565c0', '#e67e22')):
        lo, hi = cluster * n_cluster, (cluster + 1) * n_cluster
        take = (spike_indices >= lo) & (spike_indices < hi)
        ax.axhspan(lo - .5, hi - .5, color=color, alpha=.05)
        ax.scatter(spike_times[take], spike_indices[take], marker='|', s=24,
                   linewidths=.7, color=color)
    ax.axhline(n_cluster - .5, color='0.35', linestyle='--', lw=1)
    analysis_start = round(discard * sampling_frequency) / sampling_frequency
    if len(spike_times) == 0:
        ax.text(.5, .5, 'Spike events not registered', transform=ax.transAxes,
                ha='center', va='center', color='0.4')
    ax.set(xlim=(0, t_total_sim), ylim=(-.5, n_neurons - .5),
           xlabel='Time from start of simulation, s', ylabel='Neuron number')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=3,
              frameon=False)
    amp, rate = input_settings(sweep, factor)
    title = 'Permutation of amplitude' if sweep == 'amplitude' else 'Permutation of noise'
    ax.set_title(f'Spikes — {title.lower()}:'
                 f'influence_scale = {factor:g}\n'
                 f'A₁/A₂ = {amp[0]:g}/{amp[-1]:g} pA; '
                 f'r₁/r₂ = {rate[0]:g}/{rate[-1]:g} spikes/s')
    return save_plot(fig, out / f'spikes_{sweep}_seed_{simulation_seed}_scale_{factor:g}.png')


def bin_edges(centers, single_width):
    """Границы клеток по центрам, включая случай одного окна/частоты."""
    centers = np.asarray(centers, float)
    if len(centers) == 1:
        return centers[0] + np.array([-.5, .5]) * single_width
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (centers[1] - centers[0]) / 2, midpoints,
                 centers[-1] + (centers[-1] - centers[-2]) / 2]


def plot_connectivity_maps(values, freqs, simulation_seed, sweep, factor, var_order):
    """Карты обоих направлений: values[окно, метрика, направление, частота]."""
    if values.shape[0] == 0 or len(freqs) == 0:
        raise ValueError('Для карт GC/PDC/DTF нужны хотя бы одно окно и одна частота.')
    # Время отсчитывается от начала моделирования, с учётом отброшенного участка.
    times = (round(discard * sampling_frequency) +
             np.arange(values.shape[0]) * step + win_len // 2) / sampling_frequency
    time_edges = bin_edges(times, step / sampling_frequency)
    freq_edges = bin_edges(freqs, sampling_frequency / nfft)
    freq_edges = np.clip(freq_edges, fmin, fmax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    for col, metric in enumerate(METRICS):
        data = values[:, col]
        finite = data[np.isfinite(data)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if metric != 'GC':
            lo = 0.0
        if hi <= lo:
            hi = lo + 1.0
        for direction, label in enumerate(('1 → 2', '2 → 1')):
            ax = axes[direction, col]
            im = ax.pcolormesh(time_edges, freq_edges,
                               np.ma.masked_invalid(data[:, direction].T),
                               shading='flat', cmap='turbo', vmin=lo, vmax=hi)
            ax.set_title(f'{metric}, {label}')
            ax.set(xlim=(time_edges[0], time_edges[-1]),
                   ylim=(freq_edges[0], freq_edges[-1]))
            if direction == 1:
                ax.set_xlabel('Time, s')
            if col == 0:
                ax.set_ylabel('Frequency, Hz')
        # Одна и та же шкала для 1→2 и 2→1; GC сохраняет и отрицательные значения.
        fig.colorbar(im, ax=axes[:, col], shrink=.85, label=metric)
    amp, rate = input_settings(sweep, factor)
    title = 'Permutation of amplitude' if sweep == 'amplitude' else 'Permutation of noise'
    fig.suptitle(f'{title}: influence_scale = {factor:g}, '
                 f'VAR p = {var_order}\n'
                 f'A₁/A₂ = {amp[0]:g}/{amp[-1]:g} pA; '
                 f'r₁/r₂ = {rate[0]:g}/{rate[-1]:g} spikes/s')
    return save_plot(fig, out / f'gc_pdc_dtf_{sweep}_seed_{simulation_seed}_scale_{factor:g}.png')


# Поправка Holm на множественные проверки
def holm(p_values):
    p = np.asarray(p_values, float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    sort = valid[np.argsort(p[valid])]
    adjust = np.maximum.accumulate((len(sort) - np.arange(len(sort))) * p[sort])
    result[sort] = np.minimum(adjust, 1.0)
    return result

# Bootstrap по независимым seeds и максимуму |t| по частотам
def infer_spectrum(curves, rng_seed):
    """Signed absolute peak with frequency max-|t| bootstrap confidence band."""
    n = len(curves)
    if n < 5:
        raise ValueError('At least 5 complete independent seeds are required')
    mean = curves.mean(axis=0)
    se = curves.std(axis=0, ddof=1) / np.sqrt(n)
    active = se > 1e-12
    t_obs = np.divide(mean, se, out=np.zeros_like(mean), where=active)
    rng = np.random.default_rng(rng_seed)
    centered = curves - mean
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples) / n
    boot_mean = weights @ centered
    boot_second = weights @ centered ** 2
    boot_se = np.sqrt(np.maximum(boot_second - boot_mean ** 2, 0.0) / (n - 1))
    t_null = np.divide(boot_mean, boot_se, out=np.zeros_like(boot_mean), where=(boot_se > 1e-12) & active[None])
    maximum = np.max(np.abs(t_null), axis=1)
    ia = int(np.argmax(np.abs(mean)))

    def p_at(stat):
        return (1 + np.count_nonzero(maximum >= abs(stat))) / (resamples + 1)
    critical = np.quantile(maximum, 1 - alpha, method='higher')
    lower, upper = (mean - critical * se, mean + critical * se)
    return dict(D_abs_signed=float(mean[ia]), D_abs_ci_low=float(lower[ia]), D_abs_ci_high=float(upper[ia]), p_D_abs_freq=p_at(t_obs[ia]))

# Проверка параметров
def validate_config():
    if seeds < 5 or resamples < 199:
        raise ValueError('Use at least 5 seeds and 199 resamples; 30/9999 are recommended')
    if not INFLUENCE_SCALES or 1.0 not in INFLUENCE_SCALES or min(INFLUENCE_SCALES) <= 0 or (len(set(INFLUENCE_SCALES)) != len(INFLUENCE_SCALES)):
        raise ValueError('Positive, unique factors including 1.0 are required')
    if not 0 < min_valid <= 1 or not 0 < alpha < 1 or step < 1 or (dt_sim <= 0):
        raise ValueError('Invalid min_valid, alpha, step or dt')
    stride = 1 / (dt_sim * sampling_frequency)
    if abs(stride - round(stride)) > 1e-09 or stride < 1:
        raise ValueError('Recording interval must be an integer multiple of dt')
    if not np.isfinite([BASE_NOISE, SWEEP_NOISE]).all() or min(BASE_NOISE, SWEEP_NOISE) < 0:
        raise ValueError('Poisson noise rates must be finite and non-negative')
    max_rate = max(BASE_NOISE, SWEEP_NOISE * max(INFLUENCE_SCALES))
    if discard < 0 or (t_total_sim - discard) * sampling_frequency < win_len:
        raise ValueError('Not enough observations after discarding the transient')
    if not 0 <= fmin < fmax <= sampling_frequency / 2:
        raise ValueError('Invalid frequency band')
    if p_max < 1 or win_len - p_max <= 2 * p_max + 1:
        raise ValueError('Window too short for p_max')
    if order and (not 1 <= order <= p_max):
        raise ValueError('Order must be 0 or in 1..p_max')
    if PLOT_TOPOLOGY or PLOT_SPECTRA or PLOT_SPIKES:
        if PLOT_SEED_INDEX is not None and (not isinstance(PLOT_SEED_INDEX, int) or
                                           not 0 <= PLOT_SEED_INDEX < seeds):
            raise ValueError('PLOT_SEED_INDEX должен быть None или индексом в 0..seeds-1.')
    if (PLOT_SPECTRA or PLOT_SPIKES) and PLOT_FACTORS is not None:
        if np.any([factor not in INFLUENCE_SCALES for factor in PLOT_FACTORS]):
            raise ValueError('Каждый PLOT_FACTORS должен входить в INFLUENCE_SCALES.')

if __name__ == '__main__':
    validate_config()
    start_scope()
    traces = []

    # Независимые реализации; внутри seed один seed шума для всех уровней.
    for seed_index in range(seeds):
        simulation_seed = seed_start + seed_index
        signed, external = make_topology(simulation_seed)
        if PLOT_TOPOLOGY and (PLOT_SEED_INDEX is None or seed_index == PLOT_SEED_INDEX):
            plot_topology(signed, external, simulation_seed)
        values = np.empty((2, len(INFLUENCE_SCALES),
                           round(t_total_sim * sampling_frequency), 2))
        for sw, sweep in enumerate(SWEEPS):
            for fa, influence_scale in enumerate(INFLUENCE_SCALES):
                amp, rate = input_settings(sweep, influence_scale)
                record_spikes = (PLOT_SPIKES and
                                 (PLOT_SEED_INDEX is None or seed_index == PLOT_SEED_INDEX) and
                                 (PLOT_FACTORS is None or influence_scale in PLOT_FACTORS))
                result = lif_simulate(simulation_seed, signed, external, amp, rate,
                                      record_spikes=record_spikes)
                if record_spikes:
                    values[sw, fa], spike_times, spike_indices = result
                    plot_spikes(spike_times, spike_indices, simulation_seed, sweep, influence_scale)
                else:
                    values[sw, fa] = result
        traces.append(values)
        print(f'Моделирование: {seed_index + 1}/{seeds}', flush=True)

    # Общий порядок VAR по базовому уровню influence_scale = 1
    pstars = []
    base = list(INFLUENCE_SCALES).index(1.0)
    if order == 0:
        for values in traces:
            for sw in range(2):
                v = values[sw, base, round(discard * sampling_frequency):]
                try:
                    _, p_all = choose_var_order(
                        v, win_len, step, p_max=p_max, criterion='BIC',
                        ridge=ridge, aggregator='mode', verbose=False)
                    pstars.extend(p_all.tolist())
                except RuntimeError:
                    continue
        if not pstars:
            raise ValueError('Нет допустимых окон для выбора порядка VAR')
        counts = Counter(pstars)
        order = min(counts, key=lambda p: (-counts[p], p))
    print(f'Порядок VAR: p = {order}')

    freqs = np.linspace(0, sampling_frequency / 2, nfft // 2 + 1)
    freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]
    maps = []
    for seed_index, values in enumerate(traces):
        maps.append([[analyze_trace(values[sw, fa], order, freqs)
                      for fa in range(len(INFLUENCE_SCALES))] for sw in range(2)])
        if PLOT_SPECTRA and (PLOT_SEED_INDEX is None or seed_index == PLOT_SEED_INDEX):
            for sw, sweep in enumerate(SWEEPS):
                for fa, factor in enumerate(INFLUENCE_SCALES):
                    if PLOT_FACTORS is None or factor in PLOT_FACTORS:
                        plot_connectivity_maps(maps[-1][sw][fa], freqs,
                                               seed_start + seed_index, sweep, factor, order)
        print(f'Анализ: {seed_index + 1}/{seeds}', flush=True)
    maps = np.asarray(maps)  # seed, эксперимент, уровень, окно, метрика, направление, частота
    common = np.all(np.isfinite(maps[..., 0, 0, 0]), axis=2)
    fraction = common.mean(axis=-1)
    eligible = fraction >= min_valid
    delta = np.empty((seeds, 2, len(INFLUENCE_SCALES), 3, len(freqs)))
    for seed_index in range(seeds):
        for sw in range(2):
            selected = np.take(maps[seed_index, sw], np.flatnonzero(common[seed_index, sw]), axis=1)
            summed = selected.sum(axis=1)
            delta[seed_index, sw] = summed[:, :, 0] - summed[:, :, 1]
    insufficient = []
    for sw, sweep in enumerate(SWEEPS):
        count = int(eligible[:, sw].sum())
        print(f'{sweep}: допустимых seeds {count}/{seeds}; '
              f'доля общих окон {fraction[:, sw].min():.3f}–'
              f'{fraction[:, sw].max():.3f}; порог {min_valid:.3f}')
        if count < 5:
            insufficient.append(f'{sweep}: {count}/{seeds}')
    if insufficient:
        raise ValueError('Для bootstrap нужно минимум 5 допустимых seeds: '
                         + ', '.join(insufficient)
                         + '. Проверьте min_valid и длину окна win_len.')

    # Signed peak и одновременные доверительные полосы по частотам
    results = []
    for sw, sweep in enumerate(SWEEPS):
        for fa, influence_scale in enumerate(INFLUENCE_SCALES):
            for me, metric in enumerate(METRICS):
                bootstrap_seed = seed_start + 100 * sw + 10 * fa + min(me, 1)
                result = infer_spectrum(delta[eligible[:, sw], sw, fa, me], bootstrap_seed)
                results.append(result)
    adjusted = holm([r['p_D_abs_freq'] for r in results])

    # График: строки — амплитуда и шум, столбцы — GC, PDC и DTF
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for sw, sweep in enumerate(SWEEPS):
        for me, metric in enumerate(METRICS):
            indices = [sw * len(INFLUENCE_SCALES) * 3 + fa * 3 + me
                       for fa in range(len(INFLUENCE_SCALES))]
            y = np.array([results[i]['D_abs_signed'] for i in indices])
            lo = [results[i]['D_abs_ci_low'] for i in indices]
            hi = [results[i]['D_abs_ci_high'] for i in indices]
            ax = axs[sw, me]
            ax.axhline(0, color='0.25', lw=1)
            ax.fill_between(INFLUENCE_SCALES, lo, hi, color='#1565c0', alpha=.14)
            ax.plot(INFLUENCE_SCALES, y, '-o', color='#1565c0', ms=4)
            significant = adjusted[indices] < alpha
            ax.scatter(np.array(INFLUENCE_SCALES)[significant], y[significant],
                       color='#1565c0', marker='*', s=130, zorder=4)
            ax.set_title(metric)
            ax.set_xlabel(f'Amplitude influence_scale' if sw == 0
                     else f'Noise influence_scale')
            ax.set_ylabel('$D_m$')
            ax.grid(alpha=.2)
            ax.text(.03, .97, '★p < 0.05', transform=ax.transAxes, va='top')
    save_plot(fig, out / 'delta_signed_peak.pdf')
    if SHOW_PLOTS:
        plt.show()
