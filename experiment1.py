from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg as npl
start_scope()

sampling_frequency = 100
dt_sim = 1 / sampling_frequency  # шаг интегрирования (сек)
defaultclock.dt = dt_sim * second
t_total_sim = 15   # общее время моделирования (сек)
segment_switch = 5  # время переключения драйвера (сек)
n_neurons = 150
fs = sampling_frequency
n_samples = int(fs * t_total_sim)

BASE_AMPLITUDE = 100 * pA
INFLUENCE_SCALE = 2
A = INFLUENCE_SCALE * BASE_AMPLITUDE   # сильный (источник)
B = BASE_AMPLITUDE                     # базовый (цель)
C = 0 * pA                             # фоновый (без синуса)

segment_flow = ((0, 1), (1, 2), (2, 0))
R = 80 * Mohm
f  = 10 * Hz
f2 = 20 * Hz
f3 = 30 * Hz
tau = 20 * ms
phi = 0
phi2 = np.pi / 4
phi3 = np.pi / 2
J = 1 * mV
v_threshold = -50 * mV
v_reset = -70 * mV
v_rest = -65 * mV

eqs = '''
dv/dt = (v_rest - v + R*(I_half1 + I_half2 + I_half3))/tau : volt
I_half1 = int(t < 5000*ms) * amplitude * sin(2*pi*f*t  + phi) : amp
I_half2 = int((t >= 5000*ms)  and (t < 10000*ms)) * amplitude * sin(2*pi*f2*t + phi) : amp
I_half3 = int(t >= 10000*ms) * amplitude * sin(2*pi*f3*t + phi) : amp
amplitude : amp
'''

G = NeuronGroup(
    n_neurons,
    eqs,
    threshold='v > v_threshold',
    reset='v = v_reset',
    method='euler'
)
G.v = v_rest

def cluster_slice(cluster_id):
    start = cluster_id * 50
    stop = (cluster_id + 1) * 50
    return slice(start, stop)


def set_segment_amplitude(source_cluster, target_cluster):
    G.amplitude = C
    G.amplitude[cluster_slice(source_cluster)] = A
    G.amplitude[cluster_slice(target_cluster)] = B


n_steps = int(t_total_sim * fs)  # число шагов
rate1_array = np.ones(n_steps) * Hz
rate2_array = np.ones(n_steps) * Hz
rate3_array = np.ones(n_steps) * Hz

rate1_t = TimedArray(rate1_array, dt=dt_sim*second)
rate2_t = TimedArray(rate2_array, dt=dt_sim*second)
rate3_t = TimedArray(rate3_array, dt=dt_sim*second)

n_half = n_neurons // 2  # 75

P1 = PoissonGroup(n_half, rates='rate1_t(t)')
P2 = PoissonGroup(n_half, rates='rate2_t(t)')
P3 = PoissonGroup(n_half, rates='rate3_t(t)')

syn1 = Synapses(P1, G[:50], on_pre='v_post += J')
syn2 = Synapses(P2, G[50:100], on_pre='v_post += J')
syn3 = Synapses(P3, G[100:150], on_pre='v_post += J')
syn1.connect(p=1)
syn2.connect(p=1)
syn3.connect(p=1)

# ---------------- Внутренняя сеть (SBM) ----------------
sizes = [50, 50, 50]
probs = [
    [0.15, 0.05, 0.05],
    [0.05, 0.15, 0.05],
    [0.05, 0.05, 0.15]
]
g = nx.stochastic_block_model(sizes, probs, directed=True)

exc_idx = np.r_[0:40, 50:90, 100:140]
inh_idx = np.r_[40:50, 90:100, 140:150]
block_id = np.repeat(np.arange(3), 50)  # [0]*50 + [1]*50 + [2]*50

def block_weight_matrix(active_pairs):
    w_block = np.ones((3, 3), dtype=float)
    return w_block

edges = np.array(g.edges(), dtype=int)   # (m,2), исходные направленные ребра
pre = edges[:, 0]
post = edges[:, 1]
bpre  = block_id[pre]
bpost = block_id[post]
is_exc = np.isin(pre, exc_idx)
pre_exc, post_exc = pre[is_exc], post[is_exc]
pre_inh, post_inh = pre[~is_exc], post[~is_exc]

S_E = Synapses(G, G, model='w : 1', on_pre='v_post += J * w')
S_E.connect(i=pre_exc, j=post_exc)
S_I = Synapses(G, G, model='w : 1', on_pre='v_post -= J * w')
S_I.connect(i=pre_inh, j=post_inh)

def set_segment_weights(active_pairs):
    w_block = block_weight_matrix(active_pairs)
    w_all = w_block[bpre, bpost]
    S_E.w = w_all[is_exc]
    S_I.w = w_all[~is_exc]

mon = StateMonitor(G, 'v', record=True)
spike_monitor = SpikeMonitor(G)

def count_block(pre, post, block_id, a, b):
    m = (block_id[pre]==a) & (block_id[post]==b)
    return int(np.sum(m))

fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
axs = np.atleast_1d(axs)
ims = []  # чтобы потом сделать общий colorbar
for seg_idx, (src_cluster, tgt_cluster) in enumerate(segment_flow):
    set_segment_weights(((src_cluster, tgt_cluster),))
    set_segment_amplitude(src_cluster, tgt_cluster)
    W = np.zeros((n_neurons, n_neurons))
    W[S_E.i[:], S_E.j[:]] = S_E.w[:]
    W[S_I.i[:], S_I.j[:]] = S_I.w[:]
    ax = axs[seg_idx]
    im = ax.matshow(W, cmap='viridis', vmin=0, vmax=1)  # фиксируем шкалу, чтобы сравнение было честным
    ims.append(im)
    ax.set_xlabel('Postsynaptic neuron', fontsize=16)
    ax.set_ylabel('Presynaptic neuron', fontsize=16)
    ax.set_xticks(np.arange(0, 151, 50))
    ax.set_yticks(np.arange(0, 151, 50))
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)
    run(segment_switch * second)

cbar = fig.colorbar(ims[-1], ax=axs.ravel().tolist(), shrink=0.85, pad=0.02)
cbar.set_label('Synaptic weight', fontsize=16)
cbar.set_ticks(np.arange(0, 1.1, 1))
cbar.ax.tick_params(labelsize=16)
fig.savefig('weights_matrix_3segments_ru.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

spike_times = spike_monitor.t / second
spike_indices = spike_monitor.i

t_sim = mon.t / second
v1 = mon.v[:50, :].mean(axis=0) / mV
v2 = mon.v[50:100, :].mean(axis=0) / mV
v3 = mon.v[100:150, :].mean(axis=0) / mV

v = np.stack([v1, v2, v3], axis=-1)  # (T,3)
print(v.shape)

def fit_var_ols_const(y, p, ridge=0.0):
    """
    y : (T, k)
    p : порядок VAR
    Возвращает A (p,k,k), Sigma (k,k)
    """
    y = np.asarray(y, float)
    T, k = y.shape
    N = T - p
    if N <= (k * p + 1):
        raise ValueError("Слишком короткое окно для заданного p.")
    Y = y[p:]
    lags = [y[p-l-1:T-l-1] for l in range(p)]
    X = np.hstack([np.ones((N, 1), float), *lags])
    XtX = X.T @ X
    XtY = X.T @ Y
    if ridge > 0:
        B = npl.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    else:
        B = npl.solve(XtX, XtY)
    A = np.stack([B[1 + i*k : 1 + (i+1)*k, :].T for i in range(p)], axis=0)
    E = Y - X @ B
    Sigma = (E.T @ E) / N
    return A, Sigma

def freq_var_matrix(A_coef, freqs, fs):
    """
    A_coef : (p,k,k)
    freqs  : (nF,)
    Возвращает:
       A(f) : (nF,k,k)
    """
    p, k, _ = A_coef.shape
    nF = len(freqs)
    A_freq = np.empty((nF, k, k), dtype=complex)
    I_k = np.eye(k, dtype=complex)

    for idx, f in enumerate(freqs):
        z = np.exp(-1j * 2*np.pi * f / fs)
        Az = I_k.copy()
        for lag in range(p):
            Az -= A_coef[lag] * z**(lag+1)
        A_freq[idx] = Az
    return A_freq

def pdc_from_A(A_freq):
    """
    A_freq : (nF,k,k)
    Возвращает PDC: (nF,k,k), элемент [i,j] = влияние j → i.
    """
    nF, k, _ = A_freq.shape
    pdc = np.zeros((nF, k, k))

    for l in range(nF):
        Af = A_freq[l]
        denom = np.sqrt(
            np.sum(np.abs(Af)**2, axis=0, keepdims=True)
        )
        pdc[l] = np.divide(
            np.abs(Af),
            denom,
            out=np.zeros_like(np.abs(Af)),
            where=denom > 0
        )

    return pdc

win_len = 100
step    = 16
order   = 4
nfft    = 128
freqs   = np.linspace(0, fs/2, nfft//2 + 1)
n_freq  = len(freqs)

def slide_multivar(v, win_len, step, order, freqs, fs):
    """
    v : (T, k)
    Возвращает:
      PDC  : (nF,k,k,n_win)
      times: (n_win,) — центры окон
    """
    T_total, k = v.shape
    nF = len(freqs)
    starts = list(range(0, T_total - win_len + 1, step))
    n_win = len(starts)
    PDC = np.zeros((nF, k, k, n_win))
    times = np.empty(n_win)

    for w, start in enumerate(starts):
        seg = v[start:start + win_len]  # (win_len, k)
        A_full, _ = fit_var_ols_const(seg, order)
        A_freq = freq_var_matrix(A_full, freqs, fs)
        PDC[:, :, :, w] = pdc_from_A(A_freq)
        times[w] = (start + win_len//2) / fs

    return PDC, times

PDC, times = slide_multivar(v, win_len, step, order, freqs, fs)

n_win = PDC.shape[3]
dt = step / fs
time_edges = np.arange(n_win + 1) * dt   # (n_win+1,)

df = freqs[1] - freqs[0]
freq_edges = np.concatenate(([0], freqs[:-1] + df/2, [freqs[-1] + df/2]))

def plot_measure_matrix(measure, name, filename):
    """
    measure[f, i, j, w] = влияние j→i
    Строим k×k панель, каждая панель — (f × t).
    """
    if measure.ndim != 4:
        raise ValueError(f"Ожидается measure с размерностью 4 (f,i,j,w), получено {measure.shape}")

    nF, k_i, k_j, n_win_local = measure.shape
    assert k_i == k_j
    k = k_i

    fig, axs = plt.subplots(
        k, k,
        figsize=(4*k, 3*k),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axs = np.atleast_2d(axs)

    for src in range(k):      # источник (j)
        for tgt in range(k):  # приёмник (i)
            m_src_tgt = measure[:, tgt, src, :]  # (nF, n_win)
            m_src_tgt = np.nan_to_num(
                m_src_tgt,
                nan=0.0, posinf=0.0, neginf=0.0
            )

            ax = axs[src, tgt]
            im = ax.pcolormesh(
                time_edges, freq_edges, m_src_tgt,
                shading='auto', cmap='turbo'
            )
            ax.set_title(f'{name}: {src+1} → {tgt+1}', fontsize=20)

            if src == k-1:
                ax.set_xlabel('t, s', fontsize=20)
            if tgt == 0:
                ax.set_ylabel('f, Hz', fontsize=20)

            ax.set_ylim(0, fs / 2)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(),
                        shrink=0.9, pad=0.02)
    cbar.set_label(f'{name}', fontsize=20)
    cbar.ax.tick_params(labelsize=20)  # <-- размер цифр на шкале
    fig.savefig(filename, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

plot_measure_matrix(PDC, 'PDC', 'PDC_matrix_ru.pdf')