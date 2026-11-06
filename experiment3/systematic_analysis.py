"""Sweep amplitude/noise/delay and save only the six GC/PDC/DTF direction maps.

Use --report-only to redraw maps from manifest.json and runs.csv without Brian2.
"""

from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

if __package__:
    from .systematic_reporting import METRICS, plot_direction_maps
else:
    from systematic_reporting import METRICS, plot_direction_maps


BASE = Path(__file__).resolve().parent
SCHEMA = 8
HARMONIC_DRIVE = dict(waveform="harmonic", amplitude_convention="peak", phase_rad=0.0)
ROW_FIELDS = (
    "amplitude_pA", "noise_hz", "delay_ms", "seed", "metric",
    "outcome_code", "band_outcome_code",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def classify(significant):
    """Classify measured significance without looking at the imposed delay."""
    forward, reverse = np.asarray(significant, dtype=bool)
    return int(forward) + 2 * int(reverse)


def make_plan(base, amplitudes, noises, delays_ms, seeds, analysis, provenance, *, backend="numpy"):
    """One connected simulation for each grid cell and seed."""
    jobs = []
    for amplitude in amplitudes:
        for noise in noises:
            for delay in delays_ms:
                for seed in seeds:
                    config = replace(
                        base, rhythm_amplitude=float(amplitude), noise=float(noise),
                        coupling_delay=float(delay)/1000, seed=int(seed),
                    )
                    metadata = dict(
                        schema=SCHEMA, config=asdict(config), analysis=analysis,
                        source_hashes=provenance, backend=backend,
                        drive=HARMONIC_DRIVE, burn_in_s=0.0,
                    )
                    identity = hashlib.sha256(canonical_json(metadata).encode()).hexdigest()[:20]
                    jobs.append(dict(id=identity, metadata=metadata, delay_ms=float(delay)))
    return jobs


def worker_setup(cache_dir):
    import brian2 as b2
    b2.prefs.codegen.runtime.cython.cache_dir = str(cache_dir)


def run_job(job, output_dir):
    """Cache just the two sets of direction classifications for each metric."""
    if __package__:
        from .population_connectivity import analyze_population_connectivity
        from .spike_population_brian2 import LIFConfig, simulate_lif_populations
    else:
        from population_connectivity import analyze_population_connectivity
        from spike_population_brian2 import LIFConfig, simulate_lif_populations

    path = Path(output_dir) / "runs" / (job["id"] + ".npz")
    metadata = canonical_json(job["metadata"])
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["metadata_json"]) != metadata:
                raise ValueError(f"Cache metadata mismatch: {path}")
        return job["id"]

    gc.collect()  # Reuse Brian2 object names and compiled code between jobs.
    config = LIFConfig(**job["metadata"]["config"])
    counts = simulate_lif_populations(config, backend=job["metadata"]["backend"])
    analysis = job["metadata"]["analysis"]
    classes = np.full(3, 4, dtype=np.int8)
    band_classes = np.full(3, 4, dtype=np.int8)
    error = ""
    try:
        result = analyze_population_connectivity(
            counts, config.fs, p_max=analysis["p_max"],
            order_criterion=analysis["criterion"], n_surrogates=analysis["surrogates"],
            alpha=analysis["alpha"], seed=analysis["surrogate_seed"],
        )
        band = ((result.frequencies >= analysis["band_hz"][0]) &
                (result.frequencies <= analysis["band_hz"][1]))
        for index, metric in enumerate(METRICS):
            classes[index] = classify(result.significant[metric])
            band_classes[index] = classify(
                result.measures[metric][:, band].max(axis=1) > result.thresholds[metric]
            )
    except (ValueError, np.linalg.LinAlgError) as exception:
        error = str(exception)

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata_json=metadata, classes=classes,
                            band_classes=band_classes, error=error)
    temporary.replace(path)
    return job["id"]


def load_rows(output, jobs):
    rows = []
    for job in jobs:
        path = output / "runs" / (job["id"] + ".npz")
        with np.load(path, allow_pickle=False) as data:
            if str(data["metadata_json"]) != canonical_json(job["metadata"]):
                raise ValueError(f"Cache metadata mismatch: {path}")
            config = job["metadata"]["config"]
            for index, metric in enumerate(METRICS):
                rows.append(dict(
                    amplitude_pA=config["rhythm_amplitude"], noise_hz=config["noise"],
                    delay_ms=job["delay_ms"], seed=config["seed"],
                    metric=metric, outcome_code=int(data["classes"][index]),
                    band_outcome_code=int(data["band_classes"][index]),
                ))
    return rows


def read_rows(path):
    """Keep noise units from the CSV, including legacy Gaussian-noise results."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        noise_field = "noise_hz" if "noise_hz" in reader.fieldnames else "noise_mV"
        fields = tuple(noise_field if key == "noise_hz" else key for key in ROW_FIELDS)
        return [
            {key: row[key] if key == "metric" else
             int(row[key]) if key in ("seed", "outcome_code", "band_outcome_code") else
             float(row[key]) for key in fields}
            for row in reader
        ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0, 50, 100, 200],
                        help="Peak amplitudes of the harmonic current, pA.")
    parser.add_argument("--noises", type=float, nargs="+", default=[0, 10, 50, 100],
                        help="Rate of each external Poisson source, Hz.")
    parser.add_argument("--delays-ms", type=float, nargs="+", default=[-80, -40, -20, 0, 20, 40, 80])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--weight", type=float, default=3,
                        help="Absolute internal synaptic voltage jump, mV.")
    parser.add_argument("--external-probability", type=float, default=0.3)
    parser.add_argument("--external-weight", type=float, default=3,
                        help="External Poisson voltage jump, mV.")
    parser.add_argument("--excitatory-fraction", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--neurons", type=int, default=50)
    parser.add_argument("--surrogates", type=int, default=199)
    parser.add_argument("--p-max", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--output-dir", type=Path, default=BASE / "experiment3_results")
    parser.add_argument("--report-only", action="store_true",
                        help="Redraw only the six maps using saved manifest.json and runs.csv.")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if args.report_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plot_direction_maps(output, manifest, read_rows(output / "runs.csv"))
        print(f"Six direction maps: {output}", flush=True)
        return

    if __package__:
        from .spike_population_brian2 import LIFConfig, _validate
    else:
        from spike_population_brian2 import LIFConfig, _validate

    for name in ("amplitudes", "noises"):
        if any(not np.isfinite(value) or value < 0 for value in getattr(args, name)):
            parser.error(f"{name} must be finite and nonnegative")
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use positive workers and distinct nonnegative seeds.")
    if any(not np.isfinite(value) for value in args.delays_ms):
        parser.error("Delays must be finite.")
    if any(len(set(values)) != len(values) for values in (args.amplitudes, args.noises, args.delays_ms)):
        parser.error("Grid levels must be distinct.")
    if args.p_max < 1 or args.surrogates < 59:
        parser.error("Use p-max >= 1 and at least 59 surrogates for alpha=0.05 / 3.")
    base = LIFConfig(n_neurons=args.neurons, duration=args.duration, synaptic_weight=args.weight,
                     excitatory_fraction=args.excitatory_fraction,
                     external_connection_probability=args.external_probability,
                     external_weight=args.external_weight)
    try:
        for delay in args.delays_ms:
            for noise in args.noises:
                _validate(replace(base, coupling_delay=delay/1000, noise=noise))
    except ValueError as error:
        parser.error(str(error))
    analysis = dict(
        p_max=max(args.p_max, int(np.ceil(max(abs(v) for v in args.delays_ms)*base.fs/1000))+5),
        criterion="BIC", surrogates=args.surrogates, alpha=0.05, surrogate_seed=42,
        band_hz=[8.0, 12.0],
    )
    provenance = {
        name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        for name in ("systematic_analysis.py", "spike_population_brian2.py", "population_connectivity.py")
    }
    jobs = make_plan(base, args.amplitudes, args.noises, args.delays_ms, args.seeds,
                     analysis, provenance, backend=args.backend)
    manifest = dict(
        schema=SCHEMA, base_config=asdict(base), amplitudes_pA=args.amplitudes,
        noises_hz=args.noises, delays_ms=args.delays_ms, seeds=args.seeds,
        units=dict(rhythm_amplitude="pA", noise="Hz", synaptic_weight="mV",
                   coupling_delay="s", external_weight="mV",),
        poisson_mean_compensated=False, drive=HARMONIC_DRIVE, burn_in_s=0.0,
        analysis=analysis, backend=args.backend, source_hashes=provenance,
    )
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing results have different settings/code. Choose a new output directory.")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "runs").mkdir(exist_ok=True)
    missing = [job for job in jobs if not (output / "runs" / (job["id"] + ".npz")).exists()]
    print(f"{len(jobs)} simulations; {len(missing)} missing.", flush=True)
    if missing:
        cache = Path("/tmp/brian2_v3_harmonic_sweep_cython")
        cache.mkdir(exist_ok=True)
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_setup, initargs=(cache,),
        ) as pool:
            futures = [pool.submit(run_job, job, output) for job in missing]
            for completed, future in enumerate(as_completed(futures), len(jobs)-len(missing)+1):
                future.result()
                if completed % 20 == 0 or completed == len(jobs):
                    print(f"{completed}/{len(jobs)} complete.", flush=True)
    rows = load_rows(output, jobs)
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plot_direction_maps(output, manifest, rows)
    print(f"Six direction maps: {output}", flush=True)


if __name__ == "__main__":
    main()
