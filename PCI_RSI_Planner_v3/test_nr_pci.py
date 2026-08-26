"""
v3 — Teknoloji otoritesi ve PCI aralığı regresyon testleri
==========================================================
Kapsam:
  1. Teknoloji etiketi normalizasyonu (UI = tek otorite)
  2. 3GPP PCI aralıkları: LTE 0-503 (TS 36.211 §6.11),
                          NR  0-1007 (TS 38.211 §7.4.2.1)
  3. NR planlamasının 0-1007 alanını GERÇEKTEN kullanması
  4. LTE koşusunun aralık dışı PCI üretmemesi (v2'deki hata)
  5. Çıkış doğrulamasının (assert_pci_range) gerçekten tetiklenmesi
  6. NR planının kalitesi: komşuda collision yok, co-site mod3 farklı
  7. read_excel_file'ın UI teknolojisini uygulaması

Çalıştırma:  python test_nr_pci.py
"""
import io
import sys

import pandas as pd

from pci_engine import (
    norm_tech, pci_count, pci_max, sss_count, pci_in_range, assert_pci_range,
    prach_config_max,
    decompose_pci, plan_pci_network, suggest_pci, rescan_pci_rsi_for_cells,
    find_optimal_pci_rsi_for_new_cells, run_full_analysis,
    detect_sector_groups, enrich_band_columns, _is_same_site_by_id,
)
from data_handler import read_excel_file, generate_sample_excel

_fails = []


def check(cond, msg):
    if cond:
        print(f"  OK   {msg}")
    else:
        print(f"  FAIL {msg}")
        _fails.append(msg)


def build_network(n_sites=30, pci_seed=None, band='T'):
    """n_sites x 3 sektor, ~700 m aralikli grid. pci_seed(i) -> baslangic PCI."""
    rows = []
    for s in range(n_sites):
        lat = 37.7200 + (s // 6) * 0.0063      # ~700 m
        lon = 30.2800 + (s % 6) * 0.0080       # ~700 m (37.8N)
        site = f"AS{s:04d}"
        for k, (letter, az) in enumerate(zip('ADG', (0.0, 120.0, 240.0))):
            i = s * 3 + k
            rows.append({
                'cell_id': f'{band}{site}{letter}',
                'site_id': site,
                'latitude': lat, 'longitude': lon,
                'azimuth': az, 'beamwidth': 65.0,
                'pci': (pci_seed(i) if pci_seed else i % 504),
                'rsi': (i * 2) % 838,
                'prach_config_index': 3,
                'zero_correlation_zone': 5,
            })
    return enrich_band_columns(pd.DataFrame(rows))


def planned_ints(plan_df, col):
    return pd.to_numeric(plan_df[col], errors='coerce').dropna().astype(int)


# ============================================================
print("\n=== 1. Teknoloji etiketi normalizasyonu ===")
for label, expected in [('NR', 'NR'), ('nr', 'NR'), ('NR (5G)', 'NR'), ('5G', 'NR'),
                        ('5g nr', 'NR'), ('LTE', 'LTE'), ('lte', 'LTE'),
                        ('LTE (4G)', 'LTE'), ('4G', 'LTE'), ('', 'LTE'),
                        (None, 'LTE'), ('gibberish', 'LTE')]:
    check(norm_tech(label) == expected, f"norm_tech({label!r}) -> {expected}")

# ============================================================
print("\n=== 2. 3GPP PCI araliklari ===")
check(pci_count('LTE') == 504 and pci_max('LTE') == 503, "LTE: 504 PCI, 0-503")
check(pci_count('NR') == 1008 and pci_max('NR') == 1007, "NR: 1008 PCI, 0-1007")
check(sss_count('LTE') == 168, "LTE N_ID^(1): 0-167 (168 grup)")
check(sss_count('NR') == 336, "NR N_ID^(1): 0-335 (336 grup)")
# Normalizasyon aralik fonksiyonlarina da yansimali (v2'de 'nr' sessizce LTE olurdu)
check(pci_max('nr') == 1007 and pci_max('NR (5G)') == 1007,
      "kucuk harf / etiketli NR de 0-1007 veriyor")
check(pci_in_range(1007, 'NR') and not pci_in_range(1008, 'NR'), "NR sinir: 1007 gecerli, 1008 degil")
check(pci_in_range(503, 'LTE') and not pci_in_range(504, 'LTE'), "LTE sinir: 503 gecerli, 504 degil")
check(not pci_in_range(-1, 'LTE') and not pci_in_range(None, 'LTE'), "negatif / None gecersiz")
# PCI = 3*N_ID^(1) + N_ID^(2)
for p in (0, 1, 2, 503, 504, 1007):
    pss, sss = decompose_pci(p)
    check(3 * sss + pss == p and 0 <= pss <= 2, f"decompose_pci({p}) = 3*{sss}+{pss}")
check(decompose_pci(1007)[1] == 335, "NR en yuksek PCI -> N_ID^(1) = 335")
# PRACH config index araligi (TS 36.331 / TS 38.331 RACH-ConfigGeneric)
check(prach_config_max('LTE') == 63, "LTE prach-ConfigIndex: 0-63")
check(prach_config_max('NR') == 255, "NR prach-ConfigurationIndex: 0-255")
_df_pcfg = pd.DataFrame({'cell_id': ['A', 'B'], 'latitude': [37.7, 37.7],
                         'longitude': [30.2, 30.3], 'pci': [1, 2],
                         'prach_config_index': [30, 200]})
from data_handler import validate_data as _vd
_ok_lte, _err_lte = _vd(_df_pcfg, 'LTE')
_ok_nr, _err_nr = _vd(_df_pcfg, 'NR')
check(not _ok_lte and any('0-63' in e for e in _err_lte),
      "LTE'de prach-ConfigIndex 200 reddediliyor")
check(_ok_nr, "NR'de prach-ConfigurationIndex 200 kabul ediliyor")

# ============================================================
print("\n=== 3. NR planlamasi 0-1007 alanini kullaniyor mu ===")
df_nr = build_network(30, band='N')
sg_nr, c2s_nr = detect_sector_groups(df_nr)
res_nr = run_full_analysis(df_nr, 1.5, 'NR', cell_to_sector=c2s_nr)
check(res_nr['summary']['max_pci_range'] == '0-1007', "ozet PCI Araligi = 0-1007")
check(res_nr['summary']['mod6_conflict_count'] == 0 and res_nr['summary']['mod30_conflict_count'] == 0,
      "NR'de mod6/mod30 kapali (LTE'ye ozel)")

plan_nr = plan_pci_network(df_nr, res_nr['neighbors'], technology='NR',
                           sector_groups=sg_nr, cell_to_sector=c2s_nr,
                           sa_iterations_override=60_000)
pn = planned_ints(plan_nr, 'planned_pci')
check(len(pn) == len(df_nr), f"tum hucreler planlandi ({len(pn)}/{len(df_nr)})")
check(pn.min() >= 0 and pn.max() <= 1007, f"tum planli PCI 0-1007 icinde (min={pn.min()}, max={pn.max()})")
check((pn > 503).sum() > 0, f"503 ustu alan gercekten kullanildi ({(pn > 503).sum()} hucre)")
# NR'nin genis alani, LTE'ye gore daha genis bir dagilim vermeli
check(pn.nunique() > 60, f"PCI cesitliligi makul ({pn.nunique()} benzersiz deger)")
# PSS/SSS sutunlari da NR araliginda
sss_nr = planned_ints(plan_nr, 'planned_sss')
check(sss_nr.max() <= 335, f"planned_sss <= 335 (max={sss_nr.max()})")

# ============================================================
print("\n=== 4. LTE kosusu aralik disi PCI uretmemeli (v2 hatasi) ===")
# Veri NR PCI'lari tasiyor (600, 750, 1007 ...) ama kullanici LTE secti
df_mix = build_network(20, pci_seed=lambda i: (i * 37) % 1008, band='T')
sg_mix, c2s_mix = detect_sector_groups(df_mix)
check(int((df_mix['pci'] > 503).sum()) > 0,
      f"girdi verisinde {(df_mix['pci'] > 503).sum()} adet 503-ustu PCI var")
res_lte = run_full_analysis(df_mix, 1.5, 'LTE', cell_to_sector=c2s_mix)
check(res_lte['summary']['max_pci_range'] == '0-503', "ozet PCI Araligi = 0-503")

plan_lte = plan_pci_network(df_mix, res_lte['neighbors'], technology='LTE',
                            sector_groups=sg_mix, cell_to_sector=c2s_mix,
                            sa_iterations_override=60_000)
pl = planned_ints(plan_lte, 'planned_pci')
check(pl.max() <= 503, f"planlanan PCI'larin hicbiri 503'u asmiyor (max={pl.max()})")
check(pl.min() >= 0, f"planlanan PCI'lar negatif degil (min={pl.min()})")

# suggest_pci / rescan / yeni hucre de ayni garantiyi vermeli
sug = suggest_pci(df_mix, res_lte['neighbors'], res_lte, technology='LTE',
                  sector_groups=sg_mix, cell_to_sector=c2s_mix)
if len(sug) > 0:
    sp = planned_ints(sug, 'suggested_pci')
    check(len(sp) == 0 or sp.max() <= 503, f"suggest_pci ciktisi 0-503 icinde (n={len(sp)})")
else:
    print("  --   suggest_pci: onerilecek sorunlu hucre yok, atlandi")

rsc = rescan_pci_rsi_for_cells(df_mix, res_lte['neighbors'],
                               list(df_mix['cell_id'][:12]), technology='LTE',
                               sector_groups=sg_mix, cell_to_sector=c2s_mix)
if len(rsc) > 0 and 'suggested_pci' in rsc.columns:
    rp = planned_ints(rsc, 'suggested_pci')
    check(len(rp) == 0 or rp.max() <= 503, f"rescan ciktisi 0-503 icinde (n={len(rp)})")

new_cells = pd.DataFrame([{
    'cell_id': 'TAS9999A', 'site_id': 'AS9999',
    'latitude': 37.7235, 'longitude': 30.2840, 'azimuth': 60.0, 'beamwidth': 65.0,
    'pci': None, 'rsi': None, 'prach_config_index': 3, 'zero_correlation_zone': 5}])
newres = find_optimal_pci_rsi_for_new_cells(df_mix, new_cells, 1.5, technology='LTE',
                                            sector_groups=sg_mix, cell_to_sector=c2s_mix)
np_ = planned_ints(newres, 'suggested_pci')
check(len(np_) == 0 or np_.max() <= 503, f"yeni hucre onerisi 0-503 icinde (n={len(np_)})")

# Ayni veri NR olarak kosunca 503 ustu kullanilabilmeli
newres_nr = find_optimal_pci_rsi_for_new_cells(df_mix, new_cells, 1.5, technology='NR',
                                               sector_groups=sg_mix, cell_to_sector=c2s_mix)
np_nr = planned_ints(newres_nr, 'suggested_pci')
check(len(np_nr) == 0 or np_nr.max() <= 1007, "ayni veri NR modunda 0-1007 icinde")

# ============================================================
print("\n=== 5. Cikis dogrulamasi gercekten tetikleniyor mu ===")
try:
    assert_pci_range([0, 503, 504], 'LTE', 'test')
    check(False, "assert_pci_range LTE'de 504'u yakalamali")
except ValueError as e:
    check('504' in str(e) and '0-503' in str(e), f"assert_pci_range LTE hatasi: {e}")
try:
    assert_pci_range([0, 1007, 1008], 'NR', 'test')
    check(False, "assert_pci_range NR'de 1008'i yakalamali")
except ValueError as e:
    check('1008' in str(e) and '0-1007' in str(e), f"assert_pci_range NR hatasi: {e}")
try:
    assert_pci_range([0, 503, '—', None, float('nan')], 'LTE', 'test')
    check(True, "yer tutucu ('—', None, NaN) degerleri hata vermiyor")
except ValueError as e:
    check(False, f"yer tutucular hata vermemeliydi: {e}")

# ============================================================
print("\n=== 6. NR plan kalitesi ===")
pci_after = dict(zip(plan_nr['cell_id'].astype(str), planned_ints(plan_nr, 'planned_pci')))
col = m3_cosite = 0
seen = set()
for c, nbs in res_nr['neighbors'].items():
    for nb in nbs:
        key = tuple(sorted((str(c), str(nb))))
        if key in seen:
            continue
        seen.add(key)
        a, b = pci_after.get(key[0]), pci_after.get(key[1])
        if a is None or b is None:
            continue
        if c2s_nr.get(key[0]) is not None and c2s_nr.get(key[0]) == c2s_nr.get(key[1]):
            continue  # co-sektor: ayni PCI tasarim geregi
        if a == b:
            col += 1
        if _is_same_site_by_id(key[0], key[1]) and a % 3 == b % 3:
            m3_cosite += 1
check(col == 0, f"komsular arasinda PCI collision yok ({col})")
check(m3_cosite == 0, f"co-site sektorler farkli mod3 (PSS) sinifinda ({m3_cosite} ihlal)")

# ============================================================
print("\n=== 7. read_excel_file UI teknolojisini uyguluyor mu ===")
nr_bytes = generate_sample_excel('NR')
df_ok, msgs_ok = read_excel_file(io.BytesIO(nr_bytes), technology='NR')
check(df_ok is not None, "NR dosya + NR secimi -> kabul")
if df_ok is not None:
    check(set(df_ok['technology'].unique()) == {'NR'}, "technology sutunu NR olarak damgalandi")

df_bad, msgs_bad = read_excel_file(io.BytesIO(nr_bytes), technology='LTE')
check(df_bad is None, "NR dosya + LTE secimi -> reddedildi (sessizce LTE kosmuyor)")
check(any('0-503' in m for m in msgs_bad),
      f"red mesaji LTE araligini soyluyor: {[m for m in msgs_bad if '❌' in m][:1]}")

lte_bytes = generate_sample_excel('LTE')
df_l, _ = read_excel_file(io.BytesIO(lte_bytes), technology='LTE')
check(df_l is not None, "LTE dosya + LTE secimi -> kabul")
df_l_nr, msgs_l_nr = read_excel_file(io.BytesIO(lte_bytes), technology='NR')
check(df_l_nr is not None, "LTE dosya + NR secimi -> kabul (0-503, 0-1007'nin alt kumesi)")
if df_l_nr is not None:
    check(set(df_l_nr['technology'].unique()) == {'NR'},
          "UI secimi dosya etiketini eziyor (LTE etiketli dosya NR olarak damgalandi)")
    check(any('teknoloji etiketi' in m for m in msgs_l_nr),
          "uyusmazlik icin bilgilendirme mesaji verildi")

# ============================================================
print("\n" + "=" * 60)
if _fails:
    print(f"{len(_fails)} TEST BASARISIZ:")
    for f in _fails:
        print("  -", f)
    sys.exit(1)
print("Tum testler gecti.")
