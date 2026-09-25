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
print("\n=== 9. Co-sektor tutarsizligi tespiti ===")
# Analiz co-sektor ciftlerini "tasarim geregi ayni PCI" varsayarak atliyor ve
# hicbir sey bu varsayimin gecerli oldugunu dogrulamiyordu.  Gecerli degilse
# tutarsizlik, tasiyici korlugunde sahte bir "co-site collision" olarak
# goruluyor, tasiyici kapsami acikken ise hic gorunmuyordu.
from pci_engine import detect_cosector_inconsistency as _dci

_d = pd.DataFrame({
    'cell_id': ['CA', 'CB', 'DA', 'X1', 'X2'],
    'pci': [100, 100, 250, 7, 7],
    'rsi': [10, 10, 10, 20, 33],
    'carrier': ['AR1', 'AR2', 'AR3', 'AR1', 'AR2'],
})
_t = _dci(_d, {'SEC1': ['CA', 'CB', 'DA'], 'SEC2': ['X1', 'X2']})
_pci_rows = _t[_t['parametre'] == 'PCI'] if len(_t) else _t
check(len(_pci_rows) == 1, f"tek PCI tutarsizligi bulundu ({len(_pci_rows)})")
if len(_pci_rows):
    _r = _pci_rows.iloc[0]
    check(_r['sector'] == 'SEC1', "dogru sektor isaretlendi")
    check(_r['cogunluk'] == 100, f"cogunluk 100 ({_r['cogunluk']})")
    check(_r['sapan_hucre'] == 'DA', f"sapan hucre DA ({_r['sapan_hucre']})")
    check(_r['sapan_tasiyici'] == 'AR3', f"sapan tasiyici AR3 ({_r['sapan_tasiyici']})")
_rsi_rows = _t[_t['parametre'] == 'RSI'] if len(_t) else _t
check(len(_rsi_rows) == 1 and _rsi_rows.iloc[0]['sector'] == 'SEC2',
      "RSI tutarsizligi SEC2'de bulundu")
_ok = _dci(pd.DataFrame({'cell_id': ['A', 'B'], 'pci': [5, 5], 'rsi': [1, 1],
                         'carrier': ['AR1', 'AR2']}), {'S': ['A', 'B']})
check(len(_ok) == 0, "tutarli sektor rapor edilmiyor")

print("\n=== 10. Sektor kaymasi siniflandirmasi ===")
# Arac SUPHELENDIKLERINI listeler, dogrulama kullanicida — o yuzden desenin
# dogru siniflandirilmasi onemli: her desen farkli bir soru sordurur.
from pci_engine import detect_sector_shift as _dss

_rot = pd.DataFrame({
    'cell_id': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
    'site_id': ['S'] * 6,
    'azimuth': [0, 0, 120, 120, 240, 240],
    'pci':     [10, 20, 20, 30, 30, 10],     # ayni kume, farkli sektorlere
    'carrier': ['AR1', 'AR2'] * 3,
})
_c2s_r = {'A1': 's0', 'A2': 's0', 'B1': 's1', 'B2': 's1', 'C1': 's2', 'C2': 's2'}
_sg_r = {'s0': ['A1', 'A2'], 's1': ['B1', 'B2'], 's2': ['C1', 'C2']}


def _cmap(d):
    return dict(zip(d['cell_id'].astype(str), d['carrier']))


_t = _dss(_rot, _sg_r, _c2s_r, _cmap(_rot))
check(len(_t) == 1 and _t.iloc[0]['desen'] == 'ROTASYON',
      f"ayni PCI kumesi farkli sektorde -> ROTASYON "
      f"({_t.iloc[0]['desen'] if len(_t) else 'bulunamadi'})")

_azm = _rot.copy()
_azm['azimuth'] = [0, 10, 120, 130, 240, 250]     # katmanlarin azimutu farkli
_t2 = _dss(_azm, _sg_r, _c2s_r, _cmap(_azm))
check(len(_t2) == 1 and _t2.iloc[0]['desen'] == 'AZİMUT UYUŞMAZLIĞI',
      f"farkli azimut -> AZIMUT UYUSMAZLIGI "
      f"({_t2.iloc[0]['desen'] if len(_t2) else 'bulunamadi'})")

_ok = _rot.copy()
_ok['pci'] = [10, 10, 20, 20, 30, 30]
check(len(_dss(_ok, _sg_r, _c2s_r, _cmap(_ok))) == 0,
      "tutarli site rapor edilmiyor")

# AZIMUT VERISI SUPHELI: PCI isimlendirmeye gore TUTARLI ama ayni isim
# grubunun azimutlari ayrisiyor -> plan dogru, azimut sutunu supheli.
# Gercek veride SM6022 ve SM6299 boyle cikti ve kullanici dogruladi.
_susp = pd.DataFrame({
    'cell_id': ['XSITEA', 'YSITEA', 'XSITED', 'YSITED', 'XSITEG', 'YSITEG'],
    'site_id': ['SITE'] * 6,
    #                A/A         D/D         G/G   -> isim ayni sektor der
    'azimuth': [110, 10, 180, 110, 250, 180],   # ama azimutlar ayrisiyor
    'pci':     [451, 451, 140, 140, 462, 462],  # PCI isme gore TUTARLI
    'carrier': ['AR1', 'AR2'] * 3,
})
_c2s_s = {'XSITEA': 'a110', 'YSITEA': 'a10', 'XSITED': 'a180', 'YSITED': 'a110',
          'XSITEG': 'a250', 'YSITEG': 'a180'}
_sg_s = {'a110': ['XSITEA', 'YSITED'], 'a10': ['YSITEA'],
         'a180': ['XSITED', 'YSITEG'], 'a250': ['XSITEG']}
_t4 = _dss(_susp, _sg_s, _c2s_s, _cmap(_susp))
check(len(_t4) == 1 and _t4.iloc[0]['desen'] == 'AZİMUT VERİSİ ŞÜPHELİ',
      f"isme gore tutarli + azimut ayrisik -> AZIMUT VERISI SUPHELI "
      f"({_t4.iloc[0]['desen'] if len(_t4) else 'bulunamadi'})")
if len(_t4):
    check(_t4.iloc[0]['isimlendirmeye_göre_PCI'] == 'tutarlı',
          "kanit sutunu 'tutarli' diyor")
# ROTASYON'da isme gore de tutarsiz olmali — iki sinif karismasin
check(len(_t) and _t.iloc[0]['isimlendirmeye_göre_PCI'] != 'tutarlı',
      "ROTASYON ile AZIMUT VERISI SUPHELI ayirt ediliyor")

for _tt in (_t, _t2):
    check(bool(str(_tt.iloc[0]['kontrol']).strip()),
          "her satirda kullaniciya sorulacak somut bir kontrol var")

print("\n=== 11. build_carrier_map: bos carrier hucresi cozulmeli ===")
# Yeni hucre bulucu, 'carrier' sutunu ZATEN olan bir frame'e satir ekliyor ve
# yeni satirda o sutun bos kaliyor.  Bos hucre 'bilinmeyen' sayilirsa hucre
# hicbir komsuyla ayni tasiyicida olmaz ve KISITSIZ planlanir — sessizce.
from pci_engine import build_carrier_map as _bcm

_ex = enrich_carrier_column(enrich_band_columns(pd.DataFrame([
    {'cell_id': 'A', 'band': 3500, 'earfcn': 635332},
    {'cell_id': 'B', 'band': 1800, 'earfcn': 366000}])))
_new = pd.DataFrame([{'cell_id': 'NEW', 'earfcn': 635332}])
_m = _bcm(pd.concat([_ex, _new], ignore_index=True))
check(_m['NEW'] == 'AR635332',
      f"bos carrier hucresi earfcn'den cozuldu ({_m['NEW']})")
check(_m['NEW'] == _m['A'], "yeni hucre A ile ayni tasiyicida")
_m2 = _bcm(pd.concat([_ex, pd.DataFrame([{'cell_id': 'NOCAR'}])], ignore_index=True))
check(_m2['NOCAR'] == CARRIER_UNKNOWN, "hicbir kaynak yoksa bilinmiyor kalir")

print("\n=== 12. Her plotly_chart benzersiz bir key almali ===")
# Streamlit eleman ID'sini tur + parametrelerden uretir, dolayisiyla ayni
# grafigi iki kez cizmek StreamlitDuplicateElementId atar. Plan sonrasi blok
# 'Mevcut' histogramini kasitli olarak tekrar cizdigi icin bu kacinilmaz —
# key vermek zorunlu.
import ast as _ast
import collections as _coll

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py'),
            encoding='utf-8').read()
_missing, _keys = [], []
for _n in _ast.walk(_ast.parse(_src)):
    if (isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Attribute)
            and _n.func.attr == 'plotly_chart'):
        _k = [kw for kw in _n.keywords if kw.arg == 'key']
        if not _k:
            _missing.append(_n.lineno)
        else:
            try:
                _keys.append(_ast.literal_eval(_k[0].value))
            except Exception:
                _keys.append(f'<dyn@{_n.lineno}>')
check(not _missing, f"key'siz plotly_chart yok (satırlar: {_missing})")
_dup = [k for k, c in _coll.Counter(_keys).items() if c > 1]
check(not _dup, f"tekrar eden plotly_chart key'i yok ({_dup})")
print(f"       {len(_keys)} grafik, hepsi benzersiz")

# ============================================================
print("\n=== 13. RSI planlama: yuksek hiz, karma L_RA, cok tasiyicili sektor ===")
import numpy as _np
import pci_engine as E


def _apply(df_, plan_, col):
    d = df_.copy()
    d['cell_id'] = d['cell_id'].astype(str)
    for _, r in plan_.iterrows():
        if str(r[col]).isdigit():
            d.loc[d.cell_id == str(r['cell_id']), 'rsi'] = int(r[col])
    return d


def _ctx(df_, tech):
    sg_, c2s_ = E.detect_sector_groups(df_)
    nb_, _, _ = E.find_neighbors(df_, radius_km=5.0)
    return sg_, c2s_, nb_, E.build_carrier_map(df_)


# (a) Yuksek hiz: kok sayisi baslangica bagli.  Planlayici eskiden mevcut RSI'daki
#     sayiyi yeni RSI'da ayiriyor, kendi planinda cakisma birakiyordu.
_rng = _np.random.default_rng(0)
_hs = pd.DataFrame([{'cell_id': f'HS{s}{k}', 'site_id': f'S{s}', 'latitude': 41.0 + s*0.012,
                     'longitude': 36.0 + (s % 3)*0.012, 'azimuth': k*120, 'pci': s*3+k,
                     'rsi': int(_rng.integers(0, 838)), 'zero_correlation_zone': 8,
                     'prach_config_index': 3, 'high_speed': 'typeA', 'earfcn': 1800}
                    for s in range(8) for k in range(3)])
_sg, _c2s, _nb, _cm = _ctx(_hs, 'LTE')
_pl = E.plan_rsi_network(_hs, _nb, technology='LTE', sector_groups=_sg, cell_to_sector=_c2s, carrier_map=_cm)
_under = sum(1 for _, r in _pl.iterrows() if str(r['planned_rsi']).isdigit() and
             E._prach_params(_hs[_hs.cell_id == r['cell_id']].iloc[0].to_dict(), 'LTE',
                             rsi=int(r['planned_rsi']))['roots_needed'] > int(r['roots_needed']))
check(_under == 0, f"yuksek hiz: ayrilan kok her hucrede yeterli ({_under} yetersiz)")
_after = E.detect_rsi_collisions(_apply(_hs, _pl, 'planned_rsi'), _nb, technology='LTE',
                                 cell_to_sector=_c2s, carrier_map=_cm)
check(len(_after) == 0, f"yuksek hiz: plan sonrasi cakisma yok ({len(_after)})")
_sug = E.suggest_rsi(_hs, _nb, {'rsi_collisions': E.detect_rsi_collisions(
    _hs, _nb, technology='LTE', cell_to_sector=_c2s, carrier_map=_cm)},
    technology='LTE', sector_groups=_sg, cell_to_sector=_c2s, carrier_map=_cm)
_after = E.detect_rsi_collisions(_apply(_hs, _sug, 'suggested_rsi'), _nb, technology='LTE',
                                 cell_to_sector=_c2s, carrier_map=_cm)
check(len(_after) == 0, f"yuksek hiz: suggest_rsi sonrasi cakisma yok ({len(_after)})")

# (b) Ayni sektorde uzun (L=839) + kisa (L=139) hucre: kisa hucreye 137'den buyuk
#     RSI yazilmamali.  Eskiden uzun hucrenin RSI'i kopyalaniyordu.
_mix = []
for s in range(10):
    for k in range(3):
        base = dict(site_id=f'S{s}', latitude=41.0 + (s//4)*0.01, longitude=36.0 + (s % 4)*0.01, azimuth=k*120)
        _mix.append({**base, 'cell_id': f'L{s}{k}', 'pci': (s*3+k)*2, 'rsi': 0, 'prach_config_index': 12,
                     'zero_correlation_zone': 14, 'band': 1800, 'earfcn': 1850})
        _mix.append({**base, 'cell_id': f'T{s}{k}', 'pci': (s*3+k)*2+1, 'rsi': 0, 'prach_config_index': 158,
                     'zero_correlation_zone': 12, 'band': 3500, 'earfcn': 632628, 'msg1_scs_khz': 30})
_mix = pd.DataFrame(_mix)
_sg, _c2s, _nb, _cm = _ctx(_mix, 'NR')
_pl = E.plan_rsi_network(_mix, _nb, technology='NR', sector_groups=_sg, cell_to_sector=_c2s, carrier_map=_cm)
_short = _pl[_pl.cell_id.str.startswith('T')]
_bad = int((_short.planned_rsi.astype(int) > 137).sum())
check(_bad == 0, f"karma L_RA: kisa formatli hucreye >137 RSI atanmadi ({_bad})")
# Yeni kisa formatli hucre, kok alani (0-137) komsularca TAMAMEN dolu bir yerde:
# 11 komsu x 13 kok = 0..142, sarmayla 0..137'nin hepsi.  Dogru cevap "yer yok";
# eski kod 838 modulunu kullanip 143 oneriyordu.
_sat = pd.DataFrame([{'cell_id': f'SAT{i}', 'site_id': f'SS{i}', 'latitude': 41.0 + 0.001*(i % 4),
                      'longitude': 36.0 + 0.001*(i // 4), 'azimuth': 30*i, 'pci': 3*i,
                      'rsi': 13*i, 'prach_config_index': 158, 'zero_correlation_zone': 12,
                      'band': 3500, 'earfcn': 632628, 'msg1_scs_khz': 30} for i in range(11)])
_newc = _sat.iloc[:1].copy()
_newc['cell_id'] = ['SATNEW']; _newc['site_id'] = 'SSNEW'
_newc['latitude'] = 41.0015; _newc['longitude'] = 36.0015
_out = E.find_optimal_pci_rsi_for_new_cells(_sat, _newc, 5.0, technology='NR',
                                            use_antenna_direction=False)
_v = str(_out['suggested_rsi'].iloc[0])
check(not _v.isdigit() or int(_v) <= 137,
      f"dolu kok alaninda yeni kisa hucreye >137 RSI onerilmedi (oneri: {_v})")

# (c) Cok tasiyicili sektor: onerilen RSI sektordeki DIGER tasiyicilara da
#     kopyalanir, orada da temiz olmali.  Samsun'da eskiden 5 yeni cakisma birakiyordu.
#     X sektoru iki tasiyicida (XA, XB).  XA, A tasiyicisinda ZA ile cakisiyor.
#     A tarafinda en kucuk temiz RSI 10; ama B tarafinda YB 10-19'u kullaniyor.
#     Yalniz lideri kontrol eden eski kod 10'u XB'ye kopyalayip yeni cakisma yaratir.
_base = dict(prach_config_index=3, zero_correlation_zone=12)   # Ncs 119 -> 10 kok
_mc = pd.DataFrame([
    {**_base, 'cell_id': 'XA', 'site_id': 'X', 'latitude': 41.00, 'longitude': 36.000, 'azimuth': 0,   'pci': 1, 'rsi': 0,  'earfcn': 1850},
    {**_base, 'cell_id': 'XB', 'site_id': 'X', 'latitude': 41.00, 'longitude': 36.000, 'azimuth': 0,   'pci': 1, 'rsi': 0,  'earfcn': 3000},
    {**_base, 'cell_id': 'YA', 'site_id': 'Y', 'latitude': 41.01, 'longitude': 36.000, 'azimuth': 180, 'pci': 2, 'rsi': 50, 'earfcn': 1850},
    {**_base, 'cell_id': 'YB', 'site_id': 'Y', 'latitude': 41.01, 'longitude': 36.000, 'azimuth': 180, 'pci': 2, 'rsi': 10, 'earfcn': 3000},
    {**_base, 'cell_id': 'ZA', 'site_id': 'Z', 'latitude': 41.01, 'longitude': 36.002, 'azimuth': 180, 'pci': 3, 'rsi': 0,  'earfcn': 1850},
])
_sg, _c2s, _nb, _cm = _ctx(_mc, 'LTE')
_before = E.detect_rsi_collisions(_mc, _nb, technology='LTE', cell_to_sector=_c2s, carrier_map=_cm)
_sug = E.suggest_rsi(_mc, _nb, {'rsi_collisions': _before}, technology='LTE',
                     sector_groups=_sg, cell_to_sector=_c2s, carrier_map=_cm)
_after = E.detect_rsi_collisions(_apply(_mc, _sug, 'suggested_rsi'), _nb, technology='LTE',
                                 cell_to_sector=_c2s, carrier_map=_cm)
_P = lambda t: {tuple(sorted((str(a), str(b)))) for a, b in zip(t['cell_1'], t['cell_2'])} if len(t) else set()
check(not (_P(_after) - _P(_before)),
      f"cok tasiyicili sektor: oneriler yeni cakisma yaratmadi ({len(_P(_after) - _P(_before))})")
check(len(_after) <= len(_before),
      f"cok tasiyicili sektor: cakisma azaldi ({len(_before)} -> {len(_after)})")

# ============================================================
print("\n=== 14. Yeni hucre mevcut sektore katilinca sektorun PCI/RSI'ini almali ===")
# Samsun'da eskiden 40/40 yeni tasiyici hucre sektorunden farkli PCI ve RSI aliyordu.
_b = dict(prach_config_index=3, zero_correlation_zone=8)
_ex = pd.DataFrame([
    {**_b, 'cell_id': 'S1A', 'site_id': 'S1', 'latitude': 41.0, 'longitude': 36.0, 'azimuth': 0, 'pci': 30, 'rsi': 100, 'earfcn': 1850},
    {**_b, 'cell_id': 'S1B', 'site_id': 'S1', 'latitude': 41.0, 'longitude': 36.0, 'azimuth': 0, 'pci': 30, 'rsi': 100, 'earfcn': 3000},
    {**_b, 'cell_id': 'S2A', 'site_id': 'S2', 'latitude': 41.01, 'longitude': 36.0, 'azimuth': 180, 'pci': 31, 'rsi': 200, 'earfcn': 1850},
    {**_b, 'cell_id': 'S2C', 'site_id': 'S2', 'latitude': 41.01, 'longitude': 36.0, 'azimuth': 180, 'pci': 31, 'rsi': 200, 'earfcn': 5000},
])
_nw = pd.DataFrame([{**_b, 'cell_id': 'S1C', 'site_id': 'S1', 'latitude': 41.0, 'longitude': 36.0,
                     'azimuth': 0, 'pci': 0, 'rsi': 0, 'earfcn': 5000}])
_o = E.find_optimal_pci_rsi_for_new_cells(_ex, _nw, 5.0, technology='LTE').iloc[0]
check(str(_o['suggested_pci']) == '30', f"sektor PCI'i temizse onerilir (oneri {_o['suggested_pci']})")
check(str(_o['suggested_rsi']) == '100', f"sektor RSI'i temizse onerilir (oneri {_o['suggested_rsi']})")
# Ayni tasiyicida (5000) komsu S2C sektorun PCI ve RSI'ini kullaniyorsa kabul edilmemeli
_ex2 = _ex.copy()
_ex2.loc[_ex2.cell_id == 'S2C', ['pci', 'rsi']] = [30, 100]
_o = E.find_optimal_pci_rsi_for_new_cells(_ex2, _nw, 5.0, technology='LTE').iloc[0]
check(str(_o['suggested_pci']) != '30', f"sektor PCI'i yeni tasiyicida cakisiyorsa onerilmez ({_o['suggested_pci']})")
check(str(_o['suggested_rsi']) != '100', f"sektor RSI'i yeni tasiyicida cakisiyorsa onerilmez ({_o['suggested_rsi']})")
# Mevcut hucrelerin PCI'i baska bir tasiyicida ayni olsa bile (tasiyici kapsami, K-1) engel degil
_ex3 = _ex.copy()
_ex3.loc[_ex3.cell_id == 'S2A', 'pci'] = 30          # 1850'de, yeni hucre 5000'de
_o = E.find_optimal_pci_rsi_for_new_cells(_ex3, _nw, 5.0, technology='LTE').iloc[0]
check(str(_o['suggested_pci']) == '30',
      f"baska tasiyicidaki ayni PCI yeni hucreyi engellemez (oneri {_o['suggested_pci']})")

# ============================================================
print("\n=== 15. RSI stratejisi: en uzak yeniden kullanim (O-2) ===")
# 40 saha x 3 sektor, tek tasiyici, zcz 12 -> 10 kok: 1200 kok > 838, yeniden
# kullanim zorunlu.  Iki strateji de cakismasiz olmali; max_reuse'un en yakin
# yeniden kullanimi first_fit'inkinden kotu olmamali.
_g = pd.DataFrame([{'cell_id': f'G{s:02d}{k}', 'site_id': f'G{s:02d}',
                    'latitude': 41.0 + (s // 8) * 0.03, 'longitude': 36.0 + (s % 8) * 0.03,
                    'azimuth': k * 120, 'pci': (s * 3 + k) % 504, 'rsi': 0,
                    'prach_config_index': 3, 'zero_correlation_zone': 12, 'earfcn': 1850}
                   for s in range(40) for k in range(3)])
_sg, _c2s, _nb, _cm = _ctx(_g, 'LTE')
_mins = {}
for _st in ('first_fit', 'max_reuse'):
    _pl = E.plan_rsi_network(_g, _nb, technology='LTE', sector_groups=_sg,
                             cell_to_sector=_c2s, carrier_map=_cm, rsi_strategy=_st)
    _coll = E.detect_rsi_collisions(_apply(_g, _pl, 'planned_rsi'), _nb, technology='LTE',
                                    cell_to_sector=_c2s, carrier_map=_cm)
    check(len(_coll) == 0, f"{_st}: plan cakismasiz ({len(_coll)})")
    check('reuse_km' in _pl.columns and 'reuse_with' in _pl.columns,
          f"{_st}: plan ciktisinda reuse_km / reuse_with var")
    _mins[_st] = pd.to_numeric(_pl['reuse_km'], errors='coerce').min()
print(f"       en yakin yeniden kullanim: first_fit {_mins['first_fit']:.2f} km, "
      f"max_reuse {_mins['max_reuse']:.2f} km")
check(_mins['max_reuse'] >= _mins['first_fit'],
      "max_reuse'un en yakin yeniden kullanimi first_fit'ten kotu degil")
try:
    E.plan_rsi_network(_g, _nb, technology='LTE', rsi_strategy='rastgele')
    check(False, "gecersiz strateji reddedilmeli")
except ValueError:
    check(True, "gecersiz strateji ValueError verir")

# ============================================================
print("\n=== 16. Ncs yeterlilik (O-1) ===")
_b = dict(prach_config_index=3, earfcn=1850, pci=0, rsi=0, azimuth=0)
_nd = pd.DataFrame([
    {**_b, 'cell_id': 'OV1', 'latitude': 41.000, 'longitude': 36.0, 'zero_correlation_zone': 12},
    {**_b, 'cell_id': 'OV2', 'latitude': 41.018, 'longitude': 36.0, 'zero_correlation_zone': 12},  # ~2 km
    {**_b, 'cell_id': 'UN1', 'latitude': 40.000, 'longitude': 36.0, 'zero_correlation_zone': 5},
    {**_b, 'cell_id': 'UN2', 'latitude': 40.360, 'longitude': 36.0, 'zero_correlation_zone': 5},   # ~40 km
    {**_b, 'cell_id': 'NOHO', 'latitude': 39.0, 'longitude': 36.0, 'zero_correlation_zone': 5},
])
_ad = E.ncs_adequacy(_nd, {('OV1', 'OV2'): 500, ('UN1', 'UN2'): 800}, 'LTE').set_index('cell_id')
check(_ad.loc['OV1', 'status'] == 'AŞIRI', f"2 km HO, Ncs 119 (15.95 km) -> AŞIRI ({_ad.loc['OV1', 'status']})")
# ~2.0 km'yi karsilayan en kucuk: Ncs 22 (2.08 km) -> zcz 4
check(int(_ad.loc['OV1', 'suggested_zcz']) == 4 and int(_ad.loc['OV1', 'suggested_ncs']) == 22,
      f"onerilen zcz D'nin tamamini karsilayan en kucuk ({_ad.loc['OV1', 'suggested_zcz']}/"
      f"{_ad.loc['OV1', 'suggested_ncs']})")
check(float(_ad.loc['OV1', 'suggested_range_km']) >= float(_ad.loc['OV1', 'd90_km']),
      "onerilen menzil D90'i karsilar")
check(_ad.loc['UN1', 'status'] == 'YETERSİZ', f"40 km HO, Ncs 26 (2.65 km) -> YETERSİZ ({_ad.loc['UN1', 'status']})")
check(_ad.loc['NOHO', 'status'] == 'HO verisi yok', "HO verisi olmayan hucre isaretlenir")

# ============================================================
print("\n" + "=" * 60)
if _fails:
    print(f"{len(_fails)} TEST BASARISIZ:")
    for f in _fails:
        print("  -", f)
    sys.exit(1)
print("Tum testler gecti.")
