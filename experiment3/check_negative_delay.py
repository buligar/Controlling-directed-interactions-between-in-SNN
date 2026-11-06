"""Compare direct assignments of +40 ms and -40 ms to a Brian2 synapse."""

from pathlib import Path

import brian2
from brian2 import (
    Network, NeuronGroup, SpikeGeneratorGroup, SpikeMonitor, Synapses,
    defaultclock, ms, prefs, start_scope,
)
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    prefs.codegen.target = "numpy"
    n = 5
    figure, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True, sharey=True,
                                constrained_layout=True)
    saved = {}
    print(f"Brian2 {brian2.__version__}; dt = 0.1 ms; направление всегда 1 → 2")

    for column, negative in enumerate((False, True)):
        start_scope()
        defaultclock.dt = 0.1 * ms

        # The same five source spikes in both simulations; no random input.
        population1 = SpikeGeneratorGroup(n, np.arange(n), (100+5*np.arange(n)) * ms)
        population2 = NeuronGroup(
            n, "dv/dt = -v/(10*ms) : 1", threshold="v > 0.5", reset="v = 0",
            method="exact",
        )
        projection = Synapses(population1, population2, on_pre="v_post += 1")
        projection.connect(j="i")
        if negative:
            projection.delay = -40 * ms
        else:
            projection.delay = 40 * ms

        source = SpikeMonitor(population1)
        target = SpikeMonitor(population2)
        network = Network(population1, population2, projection, source, target)
        network.run(200 * ms)

        delay_ms = float(projection.delay[0] / ms)
        source_times = np.asarray(source.t / ms)
        target_times = np.asarray(target.t / ms)
        print(f"\nprojection.delay = {delay_ms:+g} ms")
        print("Популяция 1, мс:", source_times)
        print("Популяция 2, мс:", target_times)
        if len(source_times) == len(target_times):
            print("Разность времён 2 − 1, мс:", target_times-source_times)

        for row, (monitor, color) in enumerate(((source, "#2364aa"), (target, "#d14b40"))):
            axes[row, column].scatter(monitor.t/ms, monitor.i+1, marker="|", s=180,
                                       color=color)
            axes[row, column].set(ylabel=f"Популяция {row+1}\nНомер нейрона",
                                  yticks=np.arange(1, n+1), ylim=(.5, n+.5), xlim=(60, 180))
            axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(f"projection.delay = {delay_ms:+g} * ms")
        axes[1, column].set_xlabel("Время, мс")
        key = "negative" if negative else "positive"
        saved[f"{key}_source_ms"] = source_times
        saved[f"{key}_target_ms"] = target_times

    output = Path(__file__).resolve().parent / "negative_delay_check"
    output.mkdir(exist_ok=True)
    figure.suptitle(f"Одинаковый вход; связи только 1 → 2; Brian2 {brian2.__version__}")
    figure.savefig(output/"direct_delay_comparison.png", dpi=180)
    plt.close(figure)
    np.savez(output/"direct_delay_comparison.npz", **saved)
    print(f"\nГрафик: {output/'direct_delay_comparison.png'}")


if __name__ == "__main__":
    main()
