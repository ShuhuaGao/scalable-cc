"""Audit all rolling expert scenarios before imitation training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from expert_ideal10h import ROOT, SOURCE_SHA256, causal_state, generate, input_digest, extended_price


def sha(path: Path) -> str:
    """Hash one file.

    :param path: File to hash.
    :return: SHA-256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(small: bool = False) -> None:
    """Verify every manifest entry and write an audit only if all pass.

    :param small: Audit the independent small-scale suite.
    :return: None.
    """
    dest = ROOT / ('artifacts/runs/ideal10h_v1/small_experts' if small else 'artifacts/runs/ideal10h_v1/experts')
    manifest_path = dest / 'manifest.json'
    if small and not manifest_path.exists():
        manifest_path.write_text(json.dumps([{'file': f'seed_{seed}.npz', 'split': 'small', 'seed': seed, 'vehicles': 10, 'rate': 4} for seed in range(76000, 76020)], indent=2), encoding='utf8')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    expert_cfg = json.loads((ROOT / 'configs/ideal10h_v1.json').read_text(encoding='utf8'))['expert']
    archive = ROOT / 'artifacts/runs/ideal10h_v1/source_archive'
    archived_hashes = json.loads((archive / 'sha256.json').read_text(encoding='utf8'))
    if set(archived_hashes) != set(SOURCE_SHA256):
        raise ValueError('Archived source list mismatch')
    for name, expected_hash in archived_hashes.items():
        if sha(archive / name) != expected_hash:
            raise ValueError(f'Archived source hash mismatch: {name}')
    snapshot = dest / 'source_before_provenance.json'
    old_sources = json.loads(snapshot.read_text(encoding='utf8')) if snapshot.exists() else None
    entries = {}; errors = []; starts = []; old_count = 0
    nonoptimal_windows = []
    unique_durations = {}
    max_window_residual = 0.0
    max_trajectory_residual = 0.0
    max_departure_residual = 0.0
    sample_count = 0
    for entry in manifest:
        path = dest / entry['file']; report_path = path.with_suffix('.json')
        if not path.exists() or not report_path.exists():
            errors.append([entry['file'], 'missing']); continue
        try:
            report = json.loads(report_path.read_text(encoding='utf8'))
            if not report.get('complete'):
                raise ValueError(f"incomplete: {report.get('error')}")
            n = entry['vehicles']; limit = n * entry['rate']
            if report['seed'] != entry['seed'] or report['n'] != n or report['limit'] != limit or report['horizon_slots'] != 40 or report['starts'] != expert_cfg['starts'] or report['seconds_per_window'] != expert_cfg['seconds_per_10_vehicles'] * n / 10:
                raise ValueError('report configuration mismatch')
            s = generate(entry['seed'], n, fixed_day=True)
            digest = input_digest(s, limit)
            if 'schema' in report:
                if report['schema'] != 'ideal10h-perfect-information-v1' or report['input_sha256'] != digest:
                    raise ValueError('schema or input digest mismatch')
                source = report['source_sha256']
                source_provenance = 'recorded_at_generation_verified_against_archive'
                if source != archived_hashes:
                    raise ValueError('recorded expert source mismatch')
                starts.extend(w['multistart_completed'] for w in report['windows'])
            else:
                if old_sources is None:
                    raise ValueError("legacy report requires the archived batch source snapshot")
                if any(old_sources[k] != SOURCE_SHA256[k] for k in old_sources if k != "scripts/expert_mpc10h.py"):
                    raise ValueError("legacy generator or solver source mismatch")
                source = old_sources; source_provenance = "inferred_from_batch_source_snapshot"; old_count += 1
            nonoptimal_windows.extend({'file': entry['file'], 'slot': w['slot'], 'status': w['status_name'], 'residual': w['checked_residual']} for w in report['windows'] if w['status_name'] != 'KN_RC_OPTIMAL')
            max_window_residual = max(max_window_residual, max((w['checked_residual'] for w in report['windows']), default=0))
            if any(w['checked_residual'] > 1e-4 for w in report['windows']):
                raise ValueError('window residual')
            with np.load(path) as d:
                for key in ('ta', 'td', 'arrival', 'target', 'capacity', 'power', 'price'):
                    if not np.array_equal(d[key], getattr(s, key)):
                        raise ValueError(f'input mismatch: {key}')
                if not np.array_equal(d['price_extended'], extended_price(entry['seed'], s)):
                    raise ValueError('extended price mismatch')
                if float(d['limit']) != limit:
                    raise ValueError('limit mismatch')
                soc, action, x, y, records = (d[k] for k in ('soc','actions','x','y','records'))
                if soc.shape != (57,n) or action.shape != (56,n) or x.shape != (len(y),21) or records.shape != (len(y),2):
                    raise ValueError('output shape')
                if not all(np.isfinite(z).all() for z in (soc, action, x, y)):
                    raise ValueError('nonfinite output')
                active = (s.ta[None,:] <= np.arange(56)[:,None]) & (np.arange(56)[:,None] < s.td[None,:])
                expected = np.argwhere(active)
                unique_durations[(entry['split'], entry['seed'])] = s.td - s.ta
                sample_count += len(y)
                if not np.array_equal(records, expected):
                    raise ValueError('decision records')
                checks = [np.max(np.abs(soc[0]-s.arrival)), np.max(np.abs(soc[1:]-soc[:-1]-action*s.power*s.efficiency*s.dt/s.capacity)), np.max(np.maximum(-action,0)), np.max(np.maximum(action-1,0)), np.max(np.abs(action[~active]), initial=0), np.max(np.maximum(action@s.power-limit,0)), np.max(np.maximum(soc-s.target,0)), np.max(np.abs(soc[s.td,np.arange(n)]-s.target)), np.max(np.abs(y-action[records[:,0],records[:,1]]))]
                max_trajectory_residual = max(max_trajectory_residual, max(checks))
                max_departure_residual = max(max_departure_residual, float(np.max(np.abs(soc[s.td,np.arange(n)]-s.target))))
                if max(checks) > 1e-4:
                    raise ValueError(f'physical residual {max(checks)}')
                for j,(t,k) in enumerate(records):
                    if not np.allclose(x[j], causal_state(s,int(t),int(k),soc[t],action[t],limit),rtol=0,atol=1e-10):
                        raise ValueError('state mismatch')
            entries[entry['file']] = {'npz_sha256':sha(path),'report_sha256':sha(report_path),'input_sha256':digest,'source_sha256':source,'source_provenance':source_provenance}
        except Exception as exc:
            errors.append([entry['file'], str(exc)])
    durations = np.concatenate(list(unique_durations.values())) if unique_durations else np.array([])
    summary = {'nonoptimal_windows':nonoptimal_windows,'sample_count':sample_count,'departure_targets_passed':len(entries),'max_window_residual':max_window_residual,'max_trajectory_residual':max_trajectory_residual,'max_departure_residual':max_departure_residual,'unique_scenarios':len(unique_durations),'unique_vehicle_count':len(durations),'mean_parking_hours':float(np.mean(durations)*0.25) if len(durations) else None,'fraction_parking_over_10_hours':float(np.mean(durations > 40)) if len(durations) else None, 'source_archive_sha256':archived_hashes,'schema':'ideal10h-expert-audit-v1','manifest_sha256':sha(manifest_path),'entries':entries,'errors':errors,'earlier_reports_without_start_counts':old_count,'recorded_multistart_completions':{str(n):starts.count(n) for n in sorted(set(starts))}}
    target = dest / ('audit_complete.json' if not errors and len(entries) == len(manifest) else 'audit_status.json')
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf8')
    os.replace(temporary, target)
    print(f'accepted={len(entries)} errors={len(errors)} old_reports={old_count} starts={summary["recorded_multistart_completions"]}',flush=True)
    if errors:
        print(errors[:20],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true')
    audit(parser.parse_args().small)
