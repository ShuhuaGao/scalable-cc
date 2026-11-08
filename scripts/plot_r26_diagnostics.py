"""Plot fixed-cohort SoC deviation and daily action correction."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'pressure21_v1' / 'r26'
OUTPUT = ROOT / 'outputs' / 'r26_figures'


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    :param path: File to hash.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries.

    :param path: Source file.
    :return: Rows indexed by field name.
    """
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """Produce a vector figure, a preview, and a source manifest.

    :return: None.
    """
    slot_path = RESULTS / 'b_slot_daily.csv'
    daily_path = RESULTS / 'b_daily.csv'
    slot_rows = read_rows(slot_path)
    daily_rows = read_rows(daily_path)
    seeds = list(range(76000, 76020))
    slots = list(range(56))
    slot_map = {(int(r['seed']), int(r['slot'])): r for r in slot_rows}
    if len(slot_rows) != len(seeds) * len(slots) or set(slot_map) != {(s, t) for s in seeds for t in slots}:
        raise ValueError('Expected exactly 20 paired days and 56 slots per day')
    cohort = np.array([[float(slot_map[s, t]['cohort_soc_pp']) for t in slots] for s in seeds])
    if not np.isfinite(cohort).all() or (cohort < 0).any():
        raise ValueError('Invalid fixed-cohort SoC difference')
    stages = ('early', 'middle', 'late')
    stage_rows = [r for r in daily_rows if r['kind'] == 'trajectory_deviation']
    stage_map = {(int(r['seed']), r['group']): r for r in stage_rows}
    if len(stage_rows) != len(seeds) * len(stages) or set(stage_map) != {(s, g) for s in seeds for g in stages}:
        raise ValueError('Expected exactly three stage summaries for every paired day')
    correction = np.array([[float(stage_map[s, g]['correction']) for g in stages] for s in seeds])
    if not np.isfinite(correction).all() or (correction < 0).any():
        raise ValueError('Invalid stage correction')
    # Audit each daily stage mean against its active-decision-weighted slot means.
    for i, seed in enumerate(seeds):
        for j, group in enumerate(stages):
            rows = [slot_map[seed, t] for t in slots if min(2, 3 * t // len(slots)) == j]
            counts = np.array([int(row['active_count']) for row in rows])
            values = np.array([float(row['correction']) if row['correction'] else np.nan for row in rows])
            if np.any(np.isfinite(values[counts == 0])) or np.any(~np.isfinite(values[counts > 0])):
                raise ValueError(f'Inconsistent active-slot correction: {seed}, {group}')
            if not np.isclose(np.average(values[counts > 0], weights=counts[counts > 0]), correction[i, j], atol=1e-12):
                raise ValueError(f'Stage correction mismatch: {seed}, {group}')
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(6, 1.68), layout='constrained')
    mean = cohort.mean(axis=0)
    low, high = np.percentile(cohort, [25, 75], axis=0)
    axes[0].fill_between(slots, low, high, color='#d55e00', alpha=.22, linewidth=0, label='Interquartile range')
    axes[0].plot(slots, mean, color='#d55e00', lw=1.5, label='Mean')
    axes[0].set(xlabel='Slot', ylabel='SoC difference\n(percentage points)', xlim=(0, 55))
    axes[0].legend(frameon=False, loc='upper right')
    axes[1].bar(np.arange(3), correction.mean(axis=0), yerr=correction.std(axis=0, ddof=1),
                capsize=2.5, color='#d55e00', error_kw={'elinewidth': .8, 'capthick': .8})
    axes[1].set_xticks(np.arange(3), ['Early', 'Middle', 'Late'])
    axes[1].set_ylabel('Mean absolute\naction correction')
    for i, ax in enumerate(axes):
        ax.text(.02, .98, f'({chr(97+i)})', transform=ax.transAxes, ha='left', va='top')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / 'r26_drift.pdf'
    png = OUTPUT / 'r26_drift.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=250)
    plt.close(fig)
    manifest = {
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in (slot_path, daily_path)},
        'script_sha256': digest(Path(__file__)),
        'output_sha256': {'r26_drift.pdf': digest(pdf), 'r26_drift.png': digest(png)},
        'style': {'font_family': 'DejaVu Sans', 'font_size_pt': 7,
                  'data_line_width_pt': 1.5, 'axes_line_width_pt': 0.8,
                  'ours_color': '#d55e00', 'iqr_alpha': 0.22,
                  'axes_height_in': 1.2433},
        'days': seeds,
        'left': 'Each slot: fixed 10-EV cohort mean absolute closed-loop versus expert SoC difference, in percentage points; mean and interquartile range across 20 days.',
        'right': 'Each stage: per-day mean absolute raw-to-final normalized action correction over active EV-time decisions; bars are mean and whiskers sample SD across 20 days.',
        'stages': 'early slots 0-18, middle slots 19-37, late slots 38-55; zero-based',
    }
    (OUTPUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
