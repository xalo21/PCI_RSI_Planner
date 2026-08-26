"""
v2 <-> v3 A/B karsilastirma araci
=================================
Ayni hucre verisini v2 (kok klasor) ve v3 motorlariyla kosar, farklari raporlar.

Amac: "skor degisti — bir seyi duzelttik mi, yoksa bozduk mu?" sorusunu
somut olarak cevaplamak.  Her motor kendi klasorunde AYRI BIR SURECTE calisir,
boylece ayni isimli moduller birbirini ezmez.

Kullanim:
    python compare_v2_v3.py                          # sentetik ag (hizli duman testi)
    python compare_v2_v3.py --data ..\\burdur.xlsx    # gercek veri
    python compare_v2_v3.py --data x.xlsx --tech NR --radius 2.5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)

# v3'un kolon adi normalizasyonunu kullan (v2'ninkiyle ayni)
sys.path.insert(0, HERE)
from data_handler import normalize_column_names, _clean_coordinate_column  # noqa: E402


def synthetic_network(n_sites=40, seed=11, tech='LTE'):
    """Deterministik sentetik ag.  NR icin PCI 0-1007 ve hem uzun (pcfg<28)
    hem kisa (pcfg>=28) preamble kullanan hucreler uretilir — boylece
    karsilastirma NR'ye ozel yollari da kapsar."""
    rng = np.random.default_rng(seed)
    is_nr = (str(tech).upper() == 'NR')
    pci_hi = 1008 if is_nr else 504
    pcfg_set = [0, 16, 30, 40] if is_nr else [0, 3, 16, 32]
    rows = []
    for s in range(n_sites):
        lat = 37.7200 + (s // 7) * 0.0063
        lon = 30.2800 + (s % 7) * 0.0080
        site = f"AS{s:04d}"
        band = ('NH' if is_nr else 'TLEZ')[s % (2 if is_nr else 4)]
        for letter, az in zip('ADG', (0.0, 120.0, 240.0)):
            rows.append({
                'cell_id': f'{band}{site}{letter}', 'site_id': site,
                'latitude': lat, 'longitude': lon,
                'azimuth': az, 'beamwidth': 65.0,
                'pci': int(rng.integers(0, pci_hi)),
                'rsi': int(rng.integers(0, 838)),
                'prach_config_index': int(rng.choice(pcfg_set)),
                'zero_correlation_zone': int(rng.choice([5, 8, 11])),
            })
    return pd.DataFrame(rows)


def load_cells(path):
    df = pd.read_excel(path, engine='openpyxl')
    df, _, _ = normalize_column_names(df)
    for c in ('latitude', 'longitude'):
        if c in df.columns:
            df[c] = _clean_coordinate_column(df[c])
    if 'azimuth' not in df.columns:
        df['azimuth'] = 0.0
    if 'beamwidth' not in df.columns:
        df['beamwidth'] = 65.0
    if 'prach_config_index' not in df.columns:
        df['prach_config_index'] = 0
    if 'zero_correlation_zone' not in df.columns:
        df['zero_correlation_zone'] = 5
    keep = [c for c in ('cell_id', 'site_id', 'latitude', 'longitude', 'azimuth',
                        'beamwidth', 'sector', 'pci', 'rsi', 'prach_config_index',
                        'zero_correlation_zone', 'high_speed', 'cell_range')
            if c in df.columns]
    return df[keep]


def run_worker(engine_dir, csv_path, tech, radius, seed, sa_iters):
    worker = os.path.join(HERE, '_ab_worker.py')
    # PYTHONHASHSEED sabitlenmezse string hash'i her surecte farkli olur; set/dict
    # iterasyon sirasi degisir ve SA ayni tohumla bile farkli yol izler.  Sabitleyince
    # iki motor gercekten karsilastirilabilir hale gelir.
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED='0')
    # DIKKAT: `python worker.py` calistirilirsa sys.path[0] SCRIPT'in klasoru olur
    # (v3), cwd degil — o zaman iki kosu da v3 motorunu yukler ve karsilastirma
    # sessizce "her sey ayni" der.  `-c` ile calistirinca sys.path[0] = '' yani
    # cwd olur, boylece her isci kendi klasorundeki pci_engine'i yukler.
    code = f"exec(open(r'{worker}', encoding='utf-8').read())"
    proc = subprocess.run(
        [sys.executable, '-c', code, csv_path, tech, str(radius), str(seed), str(sa_iters)],
        cwd=engine_dir, capture_output=True, text=True, encoding='utf-8', env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{engine_dir} isci hatasi:\n{proc.stderr[-3000:]}")
    out = json.loads(proc.stdout)
    # Dogru motorun yuklendigini kanitla — yoksa karsilastirma anlamsiz.
    loaded = os.path.dirname(os.path.abspath(out.get('engine_file', '')))
    if loaded != os.path.abspath(engine_dir):
        raise RuntimeError(
            f"Yanlis motor yuklendi!\n  istenen : {os.path.abspath(engine_dir)}"
            f"\n  yuklenen: {loaded}")
    return out


def cmp_row(label, a, b):
    same = (a == b)
    mark = '  ' if same else '->'
    delta = ''
    if not same and isinstance(a, (int, float)) and isinstance(b, (int, float)):
        delta = f"  ({b - a:+g})"
    return f" {mark} {label:<28} v2={str(a):<12} v3={str(b):<12}{delta}", same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='Hucre verisi Excel dosyasi (yoksa sentetik ag)')
    ap.add_argument('--tech', default='LTE', choices=['LTE', 'NR'])
    ap.add_argument('--radius', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sa-iters', type=int, default=40_000)
    args = ap.parse_args()

    cells = load_cells(args.data) if args.data else synthetic_network(tech=args.tech)
    print(f"Veri     : {args.data or 'sentetik ag'}  ({len(cells)} hucre)")
    print(f"Parametre: tech={args.tech}  radius={args.radius} km  "
          f"seed={args.seed}  sa_iters={args.sa_iters:,}\n")

    tmp = os.path.join(tempfile.gettempdir(), 'ab_cells.csv')
    cells.to_csv(tmp, index=False)

    print("v2 kosuluyor...", end=' ', flush=True)
    a = run_worker(V2, tmp, args.tech, args.radius, args.seed, args.sa_iters)
    print(f"bitti   motor: {a['engine_file']}")
    print("v3 kosuluyor...", end=' ', flush=True)
    b = run_worker(HERE, tmp, args.tech, args.radius, args.seed, args.sa_iters)
    print(f"bitti   motor: {b['engine_file']}\n")

    diffs = 0

    print("=" * 74)
    print("ANALIZ OZETI")
    print("=" * 74)
    for k in a['summary']:
        line, same = cmp_row(k, a['summary'][k], b['summary'][k])
        print(line)
        diffs += (not same)

    print("\n" + "=" * 74)
    print("KOMSULUK GRAFIGI")
    print("=" * 74)
    pa, pb = set(a['neighbor_pairs']), set(b['neighbor_pairs'])
    line, same = cmp_row('komsu cifti sayisi', len(pa), len(pb))
    print(line)
    diffs += (not same)
    if pa != pb:
        print(f"    yalniz v2'de: {len(pa - pb)}   yalniz v3'te: {len(pb - pa)}")
        for p in sorted(pb - pa)[:5]:
            print(f"      + {p}")
        for p in sorted(pa - pb)[:5]:
            print(f"      - {p}")

    print("\n" + "=" * 74)
    print("HUCRE BAZLI PRACH PARAMETRELERI")
    print("=" * 74)
    fields = ('ncs', 'nzc', 'roots_needed', 'preambles_per_root',
              'cell_range_ncs_km', 'preamble_format')
    changed = {f: [] for f in fields}
    for cid, va in a['prach'].items():
        vb = b['prach'].get(cid, {})
        for f in fields:
            if va.get(f) != vb.get(f):
                changed[f].append((cid, va.get(f), vb.get(f)))
    for f in fields:
        n = len(changed[f])
        mark = '  ' if n == 0 else '->'
        print(f" {mark} {f:<28} degisen hucre: {n}")
        diffs += (n > 0)
        for cid, x, y in changed[f][:3]:
            print(f"      {cid}: {x} -> {y}")

    print("\n" + "=" * 74)
    print("PLAN CIKTILARI")
    print("=" * 74)
    for key, label in (('pci_plan', 'PCI'), ('rsi_plan', 'RSI')):
        pa_, pb_ = a.get(key, {}), b.get(key, {})
        if 'error' in pa_ or 'error' in pb_:
            print(f" -> {label}: v2={pa_.get('error', 'ok')}  v3={pb_.get('error', 'ok')}")
            diffs += 1
            continue
        for m in ('n', 'min', 'max', 'nunique', 'out_of_range'):
            if m in pa_ or m in pb_:
                line, same = cmp_row(f'{label} {m}', pa_.get(m), pb_.get(m))
                print(line)
                # SA stokastik: min/max/nunique farki beklenebilir, sayilmaz
                if m in ('n', 'out_of_range'):
                    diffs += (not same)
        ca, cb = pa_.get('by_cell', {}), pb_.get('by_cell', {})
        n_diff = sum(1 for c in ca if ca[c] != cb.get(c))
        print(f"    {label} degeri degisen hucre: {n_diff}/{len(ca)}"
              f"   (SA stokastik oldugundan fark beklenebilir)")

    print("\n" + "=" * 74)
    if diffs == 0:
        print("SONUC: v2 ve v3 deterministik ciktilarda BIREBIR AYNI.")
    else:
        print(f"SONUC: {diffs} alanda fark var (yukarida '->' ile isaretli).")
        print("Her farkin kasitli bir duzeltmeye karsilik geldigini dogrulayin.")
    print("=" * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
