"""
K-1 — Tasiyici kapsami ve planlama granularitesi testleri
=========================================================
  1. Tasiyici anahtarinin cozulmesi (earfcn > band > isimlendirme > bilinmiyor)
  2. Tasiyici bilgisi YOKSA davranis degismemeli (geriye donuk uyumluluk)
  3. Capraz tasiyici cift -> collision / mod-N / RSI raporlanmaz
  4. Confusion: belirsiz cift ayni tasiyicida olmali, ortak komsu herhangi biri
  5. split_sector_groups_by_carrier
  6. planning_scope='sector'  -> fiziksel sektor basina TEK PCI
     planning_scope='carrier' -> tasiyici basina ayri PCI

Calistirma:  python test_carrier_scope.py
"""
import sys

import pandas as pd

from pci_engine import (
    cell_carrier, build_carrier_map, enrich_carrier_column, same_carrier,
    carrier_report, split_sector_groups_by_carrier, scope_neighbors_by_carrier,
    CARRIER_UNKNOWN, PLANNING_SCOPES,
    run_full_analysis, detect_sector_groups, enrich_band_columns,
    plan_pci_network,
)

_fails = []


def check(cond, msg):
    if cond:
        print(f"  OK   {msg}")
    else:
        print(f"  FAIL {msg}")
        _fails.append(msg)


# ============================================================
print("\n=== 1. Tasiyici anahtari cozulmesi ===")
check(cell_carrier({'earfcn': 1650, 'band': 1800}) == 'AR1650',
      "earfcn, band'den once gelir")
check(cell_carrier({'arfcn': 640000}) == 'AR640000', "arfcn de kabul edilir")
check(cell_carrier({'band': 3500}) == 'B3500', "earfcn yoksa band kullanilir")
check(cell_carrier({'band_mhz': 800}) == 'B800', "band yoksa isimlendirmeden turetilen band")
check(cell_carrier({'cell_id': 'X'}) == CARRIER_UNKNOWN, "hicbiri yoksa bilinmiyor")
check(cell_carrier({'earfcn': 0, 'band': 700}) == 'B700', "earfcn=0 gecersiz sayilir")
check(cell_carrier({'earfcn': float('nan'), 'band': 700}) == 'B700', "earfcn NaN gecersiz")
# Ayni bantta iki tasiyici SADECE earfcn ile ayrilabilir
c20 = cell_carrier({'earfcn': 1650, 'band': 1800})
c10 = cell_carrier({'earfcn': 1801, 'band': 1800})
check(c20 != c10, "ayni banttaki iki tasiyici earfcn ile ayrilir")
check(cell_carrier({'band': 1800}) == cell_carrier({'band': 1800}),
      "band tek basina iki tasiyiciyi AYIRAMAZ (bilinen sinirlama)")

# ============================================================
print("\n=== 2. Tasiyici bilgisi yoksa davranis degismez ===")
check(same_carrier({}, 'A', 'B') is True, "bos harita -> her sey ayni tasiyici")
cm_unknown = {'A': CARRIER_UNKNOWN, 'B': CARRIER_UNKNOWN}
check(same_carrier(cm_unknown, 'A', 'B') is True, "iki bilinmeyen ayni kovada")
check(same_carrier({'A': 'B700', 'B': CARRIER_UNKNOWN}, 'A', 'B') is False,
      "bilinen ile bilinmeyen ayni sayilmaz")


def net(carrier_col=True):
    """2 site, 3'er sektor, her sektorde 3500 + 1800 hucresi; sektor basina tek PCI."""
    rows = []
    for si, (site, lat, lon) in enumerate((('SM0001', 41.3000, 36.3000),
                                           ('SM0002', 41.3060, 36.3000))):
        for sec, (letter, az) in enumerate(zip('ADG', (0.0, 120.0, 240.0))):
            pci = 100 + sec          # iki site AYNI PCI'lari kullaniyor -> cakisma
            for pre, band in (('C', 3500), ('D', 1800)):
                r = {'cell_id': f'{pre}{site}{letter}', 'site_id': site,
                     'latitude': lat, 'longitude': lon, 'azimuth': az,
                     'beamwidth': 65.0, 'pci': pci, 'rsi': 10 * sec,
                     'prach_config_index': 12, 'zero_correlation_zone': 12}
                if carrier_col:
                    r['band'] = band
                rows.append(r)
    return enrich_band_columns(pd.DataFrame(rows))


df = enrich_carrier_column(net())
cm = build_carrier_map(df)
n_car, n_unknown, counts = carrier_report(cm)
check(n_car == 2 and n_unknown == 0, f"2 tasiyici bulundu: {counts}")

df_nocar = net(carrier_col=False)
cm_nc = build_carrier_map(df_nocar)
check(set(cm_nc.values()) == {CARRIER_UNKNOWN},
      "band sutunu yoksa (ve isim taninmiyorsa) hepsi tek kovada")

sg, c2s = detect_sector_groups(df)
res_car = run_full_analysis(df, 2.0, 'NR', cell_to_sector=c2s)
res_nocar = run_full_analysis(df_nocar, 2.0, 'NR', cell_to_sector=c2s)
check(res_nocar['summary']['total_neighbor_pairs'] ==
      res_nocar['summary']['total_neighbor_pairs_all_layers'],
      "tasiyici bilgisi yokken tum ciftler 'ayni tasiyici' sayilir")
check(res_nocar['summary']['collision_count'] >= res_car['summary']['collision_count'],
      f"tasiyici kapsami cakisma sayisini azaltir "
      f"({res_nocar['summary']['collision_count']} -> {res_car['summary']['collision_count']})")

# ============================================================
print("\n=== 3. Capraz tasiyici cift raporlanmaz ===")
for key, label in (('collisions', 'collision'), ('mod3_conflicts', 'mod3'),
                   ('mod4_conflicts', 'mod4'), ('rsi_collisions', 'rsi')):
    t = res_car[key]
    if len(t) == 0:
        print(f"  --   {label}: tablo bos, atlandi")
        continue
    bad = [(a, b) for a, b in zip(t['cell_1'].astype(str), t['cell_2'].astype(str))
           if not same_carrier(cm, a, b)]
    check(not bad, f"{label} tablosunda capraz tasiyici satiri yok ({len(bad)} bulundu)")

print("\n=== 4. Confusion: belirsiz cift ayni tasiyicida ===")
t = res_car['confusions']
if len(t):
    bad = [(a, b) for a, b in zip(t['cell_1'].astype(str), t['cell_2'].astype(str))
           if not same_carrier(cm, a, b)]
    check(not bad, f"confusion tablosunda capraz cift yok ({len(bad)} bulundu)")
else:
    print("  --   confusion tablosu bos, atlandi")

# scope_neighbors_by_carrier
nb_scoped = scope_neighbors_by_carrier(res_car['neighbors'], cm)
bad_edges = [(c, x) for c, nbs in nb_scoped.items() for x in nbs
             if not same_carrier(cm, c, x)]
check(not bad_edges, "scope_neighbors_by_carrier capraz kenar birakmaz")

# ============================================================
print("\n=== 5. split_sector_groups_by_carrier ===")
new_sg, new_c2s = split_sector_groups_by_carrier(sg, c2s, cm)
check(len(new_sg) >= len(sg), f"grup sayisi artar veya ayni kalir ({len(sg)} -> {len(new_sg)})")
mixed = [k for k, members in new_sg.items()
         if len({cm.get(str(m), CARRIER_UNKNOWN) for m in members}) > 1]
check(not mixed, "bolunmus gruplarin hicbirinde birden fazla tasiyici yok")
check(all(cid in new_c2s for members in new_sg.values() for cid in members),
      "her uye yeni haritada var")

# ============================================================
print("\n=== 6. Planlama granularitesi ===")
check(PLANNING_SCOPES == ('sector', 'carrier'), "iki mod tanimli")
try:
    plan_pci_network(df, res_car['neighbors'], 'NR', sector_groups=sg,
                     cell_to_sector=c2s, carrier_map=cm, planning_scope='gibberish',
                     sa_iterations_override=1000)
    check(False, "gecersiz planning_scope hata vermeli")
except ValueError as e:
    check('gecersiz' in str(e), f"gecersiz planning_scope reddedildi: {e}")

import random
for scope, expect_single in (('sector', True), ('carrier', False)):
    random.seed(3)
    plan = plan_pci_network(df, res_car['neighbors'], 'NR', sector_groups=sg,
                            cell_to_sector=c2s, carrier_map=cm,
                            planning_scope=scope, sa_iterations_override=40_000)
    pmap = dict(zip(plan['cell_id'].astype(str),
                    pd.to_numeric(plan['planned_pci'], errors='coerce')))
    # Fiziksel sektor = (site, sektor harfi grubu) -> orijinal sg anahtari
    per_sector = {}
    for sec_key, members in sg.items():
        per_sector[sec_key] = {pmap.get(str(m)) for m in members if str(m) in pmap}
    n_multi = sum(1 for v in per_sector.values() if len(v) > 1)
    if expect_single:
        check(n_multi == 0,
              f"'sector' modu: her fiziksel sektor TEK PCI ({n_multi} sektorde birden fazla)")
    else:
        check(n_multi > 0,
              f"'carrier' modu: sektorler tasiyici basina ayri PCI aldi "
              f"({n_multi}/{len(per_sector)} sektor)")
    # Her iki modda da ayni tasiyicida cakisma kalmamali
    d2 = df.copy()
    d2['pci'] = d2['cell_id'].astype(str).map(pmap)
    r2 = run_full_analysis(d2, 2.0, 'NR', cell_to_sector=c2s, carrier_map=cm)
    check(r2['summary']['collision_count'] == 0,
          f"'{scope}' modu sonrasi ayni tasiyicida collision yok "
          f"({r2['summary']['collision_count']})")

# ============================================================
print("\n=== 7. app.py'deki her detect_* cagrisi carrier_map gecmeli ===")
# Bu, gercekten yasanan bir hatanin nobetcisi: plan ONCESI skor
# run_full_analysis'ten (kapsamli) gelirken, plan SONRASI skor app.py'deki
# yeniden-tespit cagrilarindan (kapsamsiz) geliyordu.  Sayilar 2.5x sisip
# skor cakiliyordu — plan degil, karsilastirma bozuktu.
import os
import re

_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
_pat = re.compile(r'detect_(?:collisions|confusions|mod3_conflicts|mod4_conflicts|'
                  r'mod6_conflicts|mod30_conflicts|rsi_collisions)\(')
_missing = []
with open(_app, encoding='utf-8') as fh:
    for i, line in enumerate(fh, 1):
        if _pat.search(line) and 'carrier_map' not in line and 'import' not in line:
            _missing.append((i, line.strip()[:80]))
check(not _missing,
      f"app.py'de carrier_map gecmeyen detect_* cagrisi yok ({len(_missing)} bulundu)")
for i, l in _missing[:5]:
    print(f"       satir {i}: {l}")

# ============================================================
print("\n=== 8. Komsuluk izgarasi hicbir cifti kacirmamali (Y-1) ===")
# Mekansal kova her iki yonde de en az yaricap kadar genis olmali. Bir boylam
# derecesi 111*cos(lat) km oldugu icin sabit 111 kullanmak dogu-bati yonunde
# cift kaciriyordu — Samsun verisinde 3 km'de 829 cift (%3.4).
import itertools as _it

import numpy as _np

from pci_engine import (find_neighbors as _fn, haversine_distance as _hd,
                        is_in_antenna_coverage as _cov)

for _lat0, _label in ((37.8, 'Burdur ~37.8N'), (41.3, 'Samsun ~41.3N'),
                      (60.0, 'yuksek enlem 60N')):
    _rng = _np.random.default_rng(3)
    _n = 220
    _la = _lat0 + _rng.random(_n) * 0.20
    _lo = 30.0 + _rng.random(_n) * 0.20
    _az = _rng.integers(0, 360, _n).astype(float)
    _d = pd.DataFrame({'cell_id': [f'G{i}' for i in range(_n)],
                       'latitude': _la, 'longitude': _lo,
                       'azimuth': _az, 'beamwidth': [65.0] * _n})
    for _R in (1.0, 3.0):
        _brute = set()
        for _i, _j in _it.combinations(range(_n), 2):
            if _hd(_la[_i], _lo[_i], _la[_j], _lo[_j]) <= _R:
                if (_cov(_la[_i], _lo[_i], _az[_i], 65.0, _la[_j], _lo[_j]) or
                        _cov(_la[_j], _lo[_j], _az[_j], 65.0, _la[_i], _lo[_i])):
                    _brute.add(tuple(sorted((f'G{_i}', f'G{_j}'))))
        _nb, _, _ = _fn(_d, _R, True, 65.0, False, None)
        _got = {tuple(sorted((str(c), str(x)))) for c, nbs in _nb.items() for x in nbs}
        _miss = _brute - _got
        check(not _miss,
              f"{_label}, R={_R} km: {len(_brute)} ciftin {len(_miss)}'i kaciyor")

# ============================================================
print("\n" + "=" * 60)
if _fails:
    print(f"{len(_fails)} TEST BASARISIZ:")
    for f in _fails:
        print("  -", f)
    sys.exit(1)
print("Tum testler gecti.")
