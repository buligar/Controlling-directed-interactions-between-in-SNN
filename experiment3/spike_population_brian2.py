"""Two LIF populations with stochastic rhythm, Poisson input and voltage jumps."""

from dataclasses import dataclass

import brian2 as b2
import numpy as np
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from scipy.signal import lfilter


# Fixed LIF dynamics and recording settings; values use seconds, mV, pA, Mohm.
INTEGRATION_DT = 0.0005
RHYTHM_BANDWIDTH = 0.65
RHYTHM_AMPLITUDE = 25.0
RESISTANCE = 80.0
TAU_M = 0.02
V_REST = -65.0
V_RESET = -70.0
V_THRESHOLD = -50.0
# Keep the existing rhythm and internal-topology stream positions.
RANDOM_STREAM_NAMES = (
    "rhythm", "poisson", "inter", "recurrent_1", "recurrent_2", "external_1", "external_2",
)


def _random_streams(seed):
    """Separate realization and purpose; never reuse seed + offset streams."""
    return dict(zip(RANDOM_STREAM_NAMES, np.random.SeedSequence(seed).spawn(len(RANDOM_STREAM_NAMES))))


@dataclass(frozen=True)
class LIFConfig:
    """Experiment controls; both external pools share one Poisson source rate."""

    n_neurons: int = 50
    duration: float = 10.0
    fs: int = 100
    frequency: float = 10.0  # Hz
    rhythm_amplitude: float = RHYTHM_AMPLITUDE  # pA; amplitude of the external rhythmic current
    coupling_delay: float = -0.04  # s; sign chooses the rhythm-driven population
    intra_connection_probability: float = 0.15
    inter_connection_probability: float = 0.05
    synaptic_weight: float = 3  # mV; approximately the previous 80 pA synapse's peak PSP
    noise: float = 50.0  # Hz per external Poisson source, equal in both populations
    seed: int = 0
    excitatory_fraction: float = 0.8  # Per population; all outgoing signs follow neuron type.
    external_connection_probability: float = 0.3
    external_weight: float = 3  # mV; excitatory external voltage jump

    @property
    def n_excitatory(self):
        return int(np.floor(self.n_neurons * self.excitatory_fraction + 0.5))

    @property
    def neuron_signs(self):
        """First E neurons, then I neurons in each population; nearest integer count."""
        return np.where(np.arange(self.n_neurons) < self.n_excitatory, 1, -1).astype(np.int8)


def _validate(config):
    for name in ("n_neurons", "fs"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 1 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(config.duration) or config.duration <= 0:
        raise ValueError("duration must be finite and positive.")
    for name in ("synaptic_weight", "noise", "rhythm_amplitude", "external_weight"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name in (
        "intra_connection_probability", "inter_connection_probability", "excitatory_fraction",
        "external_connection_probability",
    ):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if config.noise * INTEGRATION_DT >= 1:
        raise ValueError("noise * integration_dt must be < 1 for PoissonGroup input.")
    if not np.isfinite(config.seed) or config.seed < 0 or int(config.seed) != config.seed:
        raise ValueError("seed must be a nonnegative integer.")
    if not np.isfinite(config.frequency) or not 0 < config.frequency < config.fs / 2:
        raise ValueError("frequency must lie between 0 and fs / 2.")
    if not np.isfinite(config.coupling_delay) or abs(config.coupling_delay) >= config.duration:
        raise ValueError("abs(coupling_delay) must be finite and less than duration.")
    # Integer clock ratios make event binning and boundary removal unambiguous.
    ratios = (
        1 / config.fs / INTEGRATION_DT,
        config.duration * config.fs,
    )
    if any(value < 0 or not np.isclose(value, round(value), rtol=0, atol=1e-7) for value in ratios):
        raise ValueError("Recording duration and clocks must have integer sampling ratios.")
    if ratios[0] < 1:
        raise ValueError("The spike bin must not be shorter than integration_dt.")


def simulate_lif_populations(config=LIFConfig(), *, backend="numpy"):
    """Return population spike counts with shape (time bin, 2).

    Every contact is reciprocal with equal absolute weights and propagation delays.
    In each population, excitatory_fraction of neurons are excitatory (rounded to
    the nearest neuron); all others are inhibitory. Every outgoing synapse has
    the sign of its presynaptic neuron. Opposite E/I directions may have opposite
    signs; the adjacency matrix, absolute weights and delays remain symmetric.
    Positive coupling_delay applies the rhythmic input to population 1; negative
    coupling_delay applies it to population 2. Both directed
    inter-population projections exist for either sign. Brian2 receives abs(delay),
    never a negative physical propagation delay. All inter-population synapses have
    that delay in both directions. All neurons are eligible for connections.
    Postsynaptic integration can add further latency before firing. Each population
    also has reciprocal recurrent synapses without self-connections and with zero
    propagation delay. The full topology does not depend on the delay sign.

    An external stochastic 10 Hz current drives only the source. Each population has its own pool
    of n external Poisson sources, randomly connected with probability 0.3 by
    default. A source can send the same event to several neurons in its own
    population; the two pools are independent. The Poisson mean is subtracted
    from each neuron's DC current using its actual external in-degree, keeping
    the mean external drive equal to tonic_current when the source rate changes.
    There is no Gaussian membrane noise. All synapses change voltage
    instantaneously, and the reset to -70 mV has no refractory period.
    The NumPy and Cython backends run the same equations with their own
    reproducible random streams. Spike counts are retained from time zero;
    the network runs for config.duration seconds without a discarded initial interval.
    """
    _validate(config)
    if backend == "numpy":
        code_object = NumpyCodeObject
    elif backend == "cython":
        from brian2.codegen.runtime.cython_rt import CythonCodeObject
        code_object = CythonCodeObject
    else:
        raise ValueError("backend must be numpy or cython.")
    n = config.n_neurons
    streams = _random_streams(config.seed)
    total_duration = config.duration
    source = 0 if config.coupling_delay >= 0 else 1
    target = 1 - source

    delay = round(abs(config.coupling_delay) / INTEGRATION_DT) * INTEGRATION_DT

    # Input current is supplied through TimedArray; neural and synaptic dynamics
    # below are integrated by Brian2. Stationary prehistory initializes the input
    # oscillator without adding simulation time to the neural network.
    input_length = int(np.ceil(total_duration * config.fs))
    radius = np.exp(-np.pi * RHYTHM_BANDWIDTH / config.fs)
    if radius == 1:
        raise ValueError("rhythm_bandwidth is too small for a stable input oscillator.")
    oscillator = max(20 * config.fs, int(np.ceil(10 / (1 - radius))))
    innovations = np.random.default_rng(streams["rhythm"]).normal(size=input_length + oscillator)
    rhythm = lfilter(
        [1], [1, -2 * radius * np.cos(2 * np.pi * config.frequency / config.fs), radius**2],
        innovations,
    )[oscillator:]
    rhythm /= rhythm.std()
    rhythm_input = b2.TimedArray(rhythm, dt=b2.second / config.fs)
    clock = b2.Clock(dt=INTEGRATION_DT * b2.second)
    namespace = dict(
        v_rest=V_REST * b2.mV, v_reset=V_RESET * b2.mV,
        v_threshold=V_THRESHOLD * b2.mV, resistance=RESISTANCE * b2.Mohm,
        tau_m=TAU_M * b2.second,
        rhythm_input=rhythm_input,
    )
    equations = """
    dv/dt = (v_rest - v + resistance*I_drive)/tau_m : volt
    I_drive = rhythm_amplitude*rhythm_input(t) : amp
    rhythm_amplitude : amp
    """
    b2.seed(int(streams["poisson"].generate_state(1)[0]))
    populations = []
    for population in (0, 1):
        neurons = b2.NeuronGroup(
            n, equations, threshold="v > v_threshold", reset="v = v_reset",
            method="euler", clock=clock,
            namespace=namespace, codeobj_class=code_object,
            name=f"lif_population_{population + 1}*",
        )
        neurons.v = V_REST * b2.mV
        populations.append(neurons)
    populations[source].rhythm_amplitude = config.rhythm_amplitude * b2.pA

    inputs = b2.PoissonGroup(
        2*n, rates=config.noise * b2.Hz, clock=clock, codeobj_class=code_object,
        name="lif_poisson*",
    )
    external_synapses = []
    for population in (0, 1):
        external_rng = np.random.default_rng(streams[f"external_{population + 1}"])
        pre, post = np.nonzero(external_rng.random((n, n)) < config.external_connection_probability)
        if not pre.size:
            continue
        projection = b2.Synapses(
            inputs, populations[population], model="weight : volt",
            on_pre="v_post += weight", clock=clock, codeobj_class=code_object,
            name=f"lif_external_{population + 1}*",
        )
        projection.connect(i=pre + population*n, j=post)
        projection.weight = config.external_weight * b2.mV
        projection.delay = 0 * b2.second
        projection.pre.order = -2  # External events precede internal synaptic events.
        external_synapses.append(projection)
        # A voltage jump J repeated at rate K*r contributes tau_m*K*r*J
        # to the mean subthreshold voltage. Compensate using each neuron's K.
        degree = np.bincount(post, minlength=n)

    def create_synapses(source_population, target_population, pre, post, delays, name):
        if not pre.size:
            return None
        projection = b2.Synapses(
            populations[source_population], populations[target_population], model="weight : volt",
            on_pre="v_post += weight", clock=clock, codeobj_class=code_object,
            name=name,
        )
        projection.connect(i=pre, j=post)
        projection.weight = config.synaptic_weight * config.neuron_signs[pre] * b2.mV
        projection.delay = delays * b2.second
        projection.pre.order = -1
        return projection

    # Draw each population-1/population-2 pair once and create both directions.
    # The sign of the delay only chooses which population receives the rhythm.
    topology_rng = np.random.default_rng(streams["inter"])
    population1, population2 = np.nonzero(
        topology_rng.random((n, n)) < config.inter_connection_probability
    )
    pre, post = (population1, population2) if source == 0 else (population2, population1)
    synapses = create_synapses(
        source, target, pre, post, delay, "lif_projection*",
    )
    reverse_synapses = create_synapses(
        target, source, post, pre, delay, "lif_reverse_projection*",
    )
    recurrent_synapses = []
    for population in (0, 1):
        recurrent_rng = np.random.default_rng(streams[f"recurrent_{population + 1}"])
        upper = np.triu(recurrent_rng.random((n, n)) < config.intra_connection_probability, k=1)
        connected = upper | upper.T
        pre, post = np.nonzero(connected)
        recurrent_synapses.append(create_synapses(
            population, population, pre, post, 0.0, f"lif_recurrent_{population + 1}*",
        ))

    spike_monitors = tuple(
        b2.SpikeMonitor(population, codeobj_class=code_object) for population in populations
    )
    objects = [*populations, inputs, *external_synapses, *spike_monitors]
    if synapses is not None:
        objects.append(synapses)
    if reverse_synapses is not None:
        objects.append(reverse_synapses)
    objects.extend(projection for projection in recurrent_synapses if projection is not None)
    network = b2.Network(*objects)
    network.run(total_duration * b2.second, namespace={})

    n_bins = int(round(config.duration * config.fs))
    ticks_per_bin = int(round(1 / config.fs / INTEGRATION_DT))
    record_ticks = n_bins * ticks_per_bin
    counts = np.zeros((n_bins, 2), dtype=np.int32)
    for population, monitor in enumerate(spike_monitors):
        ticks = np.rint(np.asarray(monitor.t / b2.second) / INTEGRATION_DT).astype(int)
        selected = (ticks >= 0) & (ticks < record_ticks)
        ticks = ticks[selected]
        np.add.at(counts[:, population], ticks // ticks_per_bin, 1)
    return counts
