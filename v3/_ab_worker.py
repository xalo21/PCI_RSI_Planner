"""
A/B karsilastirma isci sureci
=============================
Kendi klasorundeki pci_engine'i yukler, verilen hucre CSV'si uzerinde analiz +
planlama kosar ve sonucu JSON olarak stdout'a basar.

compare_v2_v3.py bunu hem v2 hem v3 klasorunde ayri birer surec olarak calistirir
— boylece iki motor birbirinin sys.modules'unu kirletmez.

Kullanim (dogrudan cagrilmaz):
    python _ab_worker.py <cells.csv> <LTE|NR> <radius_km> <seed> <sa_iters>
"""
import json
import random
import sys

import numpy as np
import pandas as pd

import pci_engine as E


def jsonable(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def main():
    csv_path, tech, radius, seed, sa_iters = sys.argv[1:6]
    radius = float(radius)
    seed = int(seed)
    sa_iters = int(sa_iters)

    df = pd.read_csv(csv_path)
    df['cell_id'] = df['cell_id'].astype(str)
    if 'site_id' in df.columns:
        df['site_id'] = df['site_id'].astype(str)
    df = E.enrich_band_columns(df)

    random.seed(seed)
    np.random.seed(seed)

    sg, c2s = E.detect_sector_groups(df, azimuth_tolerance=10.0)
    res = E.run_full_analysis(df, radius, tech, cell_to_sector=c2s)
    s = res['summary']

    out = {
        'engine_file': E.__file__,
        'summary': {k: jsonable(s.get(k)) for k in (
            'technology', 'total_cells', 'total_neighbor_pairs', 'max_pci_range',
            'collision_count', 'confusion_count', 'mod3_conflict_count',
            'mod4_conflict_count', 'mod6_conflict_count', 'mod30_conflict_count',
            'rsi_collision_count', 'cosite_collision_count', 'cosite_mod3_count',
            'cells_with_issues', 'health_score')},
        'neighbor_pairs': sorted(
            '|'.join(sorted((str(a), str(b))))
            for a, nbs in res['neighbors'].items() for b in nbs),
    }

    # --- per-cell PRACH parameters (refactor'in asil hedefi) ---
    prach = {}
    for _, r in df.iterrows():
        info = E.compute_cell_prach_info(r, tech)
        prach[str(r['cell_id'])] = {
            'ncs': jsonable(info.get('ncs')),
            'nzc': jsonable(info.get('nzc')),
            'roots_needed': jsonable(info.get('roots_needed')),
            'preambles_per_root': jsonable(info.get('preambles_per_root')),
            'cell_range_ncs_km': jsonable(info.get('cell_range_ncs_km')),
            'preamble_format': jsonable(info.get('preamble_format')),
        }
    out['prach'] = prach

    # --- PCI plani ---
    random.seed(seed)
    np.random.seed(seed)
    try:
        plan = E.plan_pci_network(df, res['neighbors'], technology=tech,
                                  sector_groups=sg, cell_to_sector=c2s,
                                  sa_iterations_override=sa_iters)
        pp = pd.to_numeric(plan['planned_pci'], errors='coerce').dropna().astype(int)
        out['pci_plan'] = {
            'n': int(len(pp)), 'min': int(pp.min()) if len(pp) else None,
            'max': int(pp.max()) if len(pp) else None,
            'nunique': int(pp.nunique()),
            'out_of_range': int((pp > E.pci_max(tech)).sum() + (pp < 0).sum())
            if hasattr(E, 'pci_max') else
            int((pp > (1007 if tech == 'NR' else 503)).sum()),
            'by_cell': {str(c): int(p) for c, p in
                        zip(plan['cell_id'], pd.to_numeric(plan['planned_pci'],
                                                           errors='coerce').fillna(-1).astype(int))},
        }
    except Exception as e:  # planlayici patlarsa karsilastirma yine de is gorsun
        out['pci_plan'] = {'error': f'{type(e).__name__}: {e}'}

    # --- RSI plani ---
    random.seed(seed)
    np.random.seed(seed)
    try:
        rplan = E.plan_rsi_network(df, res['neighbors'], technology=tech,
                                   sector_groups=sg, cell_to_sector=c2s)
        rr = pd.to_numeric(rplan['planned_rsi'], errors='coerce').dropna().astype(int)
        out['rsi_plan'] = {
            'n': int(len(rr)), 'min': int(rr.min()) if len(rr) else None,
            'max': int(rr.max()) if len(rr) else None,
            'nunique': int(rr.nunique()),
            'by_cell': {str(c): int(v) for c, v in
                        zip(rplan['cell_id'], pd.to_numeric(rplan['planned_rsi'],
                                                            errors='coerce').fillna(-1).astype(int))},
        }
    except Exception as e:
        out['rsi_plan'] = {'error': f'{type(e).__name__}: {e}'}

    sys.stdout.write(json.dumps(out))


if __name__ == '__main__':
    main()
