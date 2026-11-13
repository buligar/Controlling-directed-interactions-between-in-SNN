"""The six GC/PDC/DTF direction maps for the Brian2 parameter sweep."""

from collections import Counter, defaultdict

import numpy as np


METRICS = ("GC", "PDC", "DTF")
OUTCOMES = (
    "Нет значимости", "Только 1 → 2", "Только 2 → 1", "Оба направления",
    "Не оценивается", "Нет единого большинства",
)


def plot_direction_maps(output, manifest, rows):
    """Use a strict majority of all seeds, including non-estimable runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    poisson = "noises_hz" in manifest
    noise_field = "noise_hz" if poisson else "noise_mV"
    amplitudes = manifest["amplitudes_pA"]
    noises = manifest["noises_hz" if poisson else "noises_mV"]
    delays = manifest["delays_ms"]
    seeds = set(manifest["seeds"])
    by_cell = defaultdict(dict)
    for row in rows:
        key = (row["amplitude_pA"], row[noise_field], row["delay_ms"], row["metric"])
        if row["seed"] in by_cell[key]:
            raise ValueError(f"Duplicate seed in cell {key}: {row['seed']}")
        by_cell[key][row["seed"]] = row
    for amplitude in amplitudes:
        for noise in noises:
            for delay in delays:
                for metric in METRICS:
                    key = (amplitude, noise, delay, metric)
                    if not seeds or set(by_cell[key]) != seeds:
                        raise ValueError(f"Incomplete seed set in cell {key}")

    colors = ("#e4e4e4", "#2878b5", "#e68123", "#9852a3", "#323232", "#e9c64c")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6, 1), cmap.N)
    for metric in METRICS:
        for band in (False, True):
            field = "band_outcome_code" if band else "outcome_code"
            figure, axes = plt.subplots(
                1, len(amplitudes), figsize=(max(7, len(amplitudes)*4), 4.8),
                squeeze=False, layout="constrained",
            )
            axes = axes[0]
            for amplitude, axis in zip(amplitudes, axes):
                axis.set_xlabel(f"coupling_delay, мс\nАмплитуда {amplitude:g} пА")
                axis.set_xticks(range(len(delays)), [f"{d:g}" for d in delays])
                axis.set_yticks(range(len(noises)), [f"{v:g}" for v in noises])
                codes = np.empty((len(noises), len(delays)), dtype=int)
                for i, noise in enumerate(noises):
                    for j, delay in enumerate(delays):
                        samples = by_cell[(amplitude, noise, delay, metric)].values()
                        counts = Counter(row[field] for row in samples)
                        winners = [code for code, count in counts.items() if count > len(samples)/2]
                        code = winners[0] if winners else 5
                        codes[i, j] = code
                        text = f"{counts[code]}/{len(samples)}" if code < 5 else "смесь"
                        axis.text(j, i, text, ha="center", va="center", fontsize=8,
                                  color="white" if code in (1, 3, 4) else "black")
                axis.imshow(codes, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            axes[0].set_ylabel("Частота пуассоновского входа, Гц" if poisson else "Шум, мВ")
            figure.legend(
                handles=[Patch(facecolor=c, label=label) for c, label in zip(colors, OUTCOMES)],
                loc="outside lower center", ncols=3, fontsize=9,
            )
            name = f"direction_10Hz_{metric}.png" if band else f"direction_{metric}.png"
            figure.savefig(output / name, dpi=160)
            plt.close(figure)
