"""
3GPP tablo spesifikasyonu — test-first
======================================
Bu dosya, motorun 3GPP tablolarina UYMASI GEREKEN davranisini tarif eder.
Bir kismi bilerek HENUZ GECMIYOR: onlar K-2 / K-3 / K-4 duzeltmelerinin
sartnamesidir.  Duzeltme yapildikca ilgili grubun expect_fail bayragi kalkar.

Kaynaklar:
  TS 36.211  Tablo 5.7.1-1  LTE preamble format parametreleri
             Tablo 5.7.1-2  PRACH config index -> format (FDD, frame type 1)
             Tablo 5.7.2-2  Ncs, format 0-3 (unrestricted + restricted set A)
             Tablo 5.7.2-3  Ncs, format 4
             Bolum 5.7.2    Kisitli kumede kok basina cevrimsel kayma sayisi
  TS 38.211  Tablo 6.3.3.1-1  NR preamble formatlari (L_RA, delta_f_RA)
             Tablo 6.3.3.1-5  Ncs, L_RA = 839
             Tablo 6.3.3.1-6  Ncs, L_RA = 139

Calistirma:  python test_3gpp_tables.py
"""
import sys
from math import ceil

import pci_engine as E

# grup adi -> (henuz gecmesi beklenmiyor mu, hangi bulgu)
GROUPS = {}
RESULTS = []


def group(name, expect_fail=False, finding=''):
    GROUPS[name] = {'expect_fail': expect_fail, 'finding': finding,
                    'pass': 0, 'fail': 0, 'msgs': []}
    return name


def spec(g, cond, msg):
    rec = GROUPS[g]
    if cond:
        rec['pass'] += 1
    else:
        rec['fail'] += 1
        rec['msgs'].append(msg)


# ============================================================
# Bagimsiz referans uygulamalar (motordan degil, standarttan)
# ============================================================
NZC = 839


def ref_du(u, nzc=NZC):
    """d_u: (p*u) mod N_ZC = 1 kosulunu saglayan p; p < N_ZC/2 ise p, degilse N_ZC-p."""
    p = pow(u, -1, nzc)
    return p if p < nzc / 2 else nzc - p


def ref_restricted_shifts(u, ncs, nzc=NZC):
    """TS 36.211 §5.7.2 / TS 38.211 §6.3.3.1 — kisitli kume (tip A),
    bir kokten uretilebilen cevrimsel kayma sayisi."""
    d = ref_du(u, nzc)
    if ncs <= d < nzc / 3:
        n_shift = d // ncs
        d_start = 2 * d + n_shift * ncs
        n_group = nzc // d_start
        n_shift_bar = max((nzc - 2 * d - n_group * d_start) // ncs, 0)
    elif nzc / 3 <= d <= (nzc - ncs) // 2:
        n_shift = (nzc - 2 * d) // ncs
        d_start = nzc - 2 * d + n_shift * ncs
        n_group = d // d_start
        n_shift_bar = min(max((d - n_group * d_start) // ncs, 0), n_shift)
    else:
        return 0
    return n_shift * n_group + n_shift_bar


def ref_window_us(delta_f_ra_khz):
    """Cevrimsel kayma penceresinin dayandigi tek dizi suresi = 1 / delta_f_RA."""
    return 1000.0 / delta_f_ra_khz


def ref_cell_range_km(ncs, l_ra, delta_f_ra_khz):
    """Bir Ncs'in destekledigi en buyuk hucre yaricapi (km).

    Sesia/Toufik/Baker, "LTE - The UMTS Long Term Evolution", 2. baski,
    denklem 17.10:  Ncs >= ceil((20/3 r + tau_ds) Nzc / Tseq) + n_g, 3GPP Ncs
    kumesi tau_ds = 5.2 us ve n_g = 2 ile tasarlanmistir (s.390).  Pay yalnizca
    dogrulandigi yerde (L_RA=839, 1.25 kHz) uygulanir; digerlerinde duz
    gidis-donus  r = Ncs / (L_RA * delta_f_RA) * c / 2.
    """
    t_seq_us = 1000.0 / delta_f_ra_khz
    if l_ra == 839 and abs(delta_f_ra_khz - 1.25) < 1e-9:
        window_us = (ncs - 2) * t_seq_us / l_ra - 5.2
    else:
        window_us = ncs * t_seq_us / l_ra
    return max(window_us, 0.0) * 1e-6 * 3e8 / 2 / 1000.0


# ============================================================
g1 = group('LTE Ncs — unrestricted (T5.7.2-2)')
EXPECTED_LTE_UNRESTRICTED = [0, 13, 15, 18, 22, 26, 32, 38, 46, 59, 76, 93,
                             119, 167, 279, 419]
for zcz, exp in enumerate(EXPECTED_LTE_UNRESTRICTED):
    spec(g1, E.get_ncs(zcz, 'LTE', restricted=False) == exp,
         f"zcz={zcz}: {E.get_ncs(zcz,'LTE')} != {exp}")

g2 = group('LTE Ncs — restricted set A/B (T5.7.2-2)')
# DIKKAT: bu testin ilk hali YANLISTI — kisitsiz sutunun bir kaydirilmisi
# (59, 76, 93, 119, 167, 279, 419, 839) yazilmisti ve kod da ayni hatayi
# tasidigi icin test geciyordu. Asagidaki degerler spec metninden cikarildi.
EXPECTED_LTE_RESTRICTED_A = [15, 18, 22, 26, 32, 38, 46, 55, 68, 82, 100, 128,
                             158, 202, 237]          # zcz=15 N/A
EXPECTED_LTE_RESTRICTED_B = [15, 18, 22, 26, 32, 38, 46, 55, 68, 82, 100, 118,
                             137]                    # zcz>=13 N/A
for zcz, exp in enumerate(EXPECTED_LTE_RESTRICTED_A):
    spec(g2, E.get_ncs(zcz, 'LTE', restricted=True) == exp,
         f"tipA zcz={zcz}: {E.get_ncs(zcz,'LTE',restricted=True)} != {exp}")
spec(g2, E.get_ncs(15, 'LTE', restricted=True) == 0, "tipA zcz=15 N/A olmali")
for zcz, exp in enumerate(EXPECTED_LTE_RESTRICTED_B):
    spec(g2, E.get_ncs(zcz, 'LTE', restricted='typeB') == exp,
         f"tipB zcz={zcz}: {E.get_ncs(zcz,'LTE',restricted='typeB')} != {exp}")
spec(g2, E.get_ncs(13, 'LTE', restricted='typeB') == 0, "tipB zcz=13 N/A olmali")

g3 = group('NR Ncs — L_RA=839 (T6.3.3.1-5)')
for zcz, exp in enumerate(EXPECTED_LTE_UNRESTRICTED):   # NR uzun dizi ile ayni
    spec(g3, E.get_ncs(zcz, 'NR', short=False) == exp,
         f"zcz={zcz}: {E.get_ncs(zcz,'NR')} != {exp}")

g4 = group('NR Ncs — L_RA=139 (T6.3.3.1-6)')
EXPECTED_NR_SHORT = [0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 23, 27, 34, 46, 69]
for zcz, exp in enumerate(EXPECTED_NR_SHORT):
    spec(g4, E.get_ncs(zcz, 'NR', short=True) == exp,
         f"zcz={zcz}: {E.get_ncs(zcz,'NR',short=True)} != {exp}")

# ============================================================
g5 = group('LTE Ncs — format 4 / L_RA=139 (T5.7.2-3)')
EXPECTED_LTE_FMT4 = {0: 2, 1: 4, 2: 6, 3: 8, 4: 10, 5: 12, 6: 15}
for zcz, exp in EXPECTED_LTE_FMT4.items():
    got = E.get_ncs(zcz, 'LTE', short=True)
    spec(g5, got == exp, f"zcz={zcz}: {got} != {exp}")
spec(g5, E.get_ncs(7, 'LTE', short=True) == 0,
     f"zcz=7 format 4'te N/A olmali, {E.get_ncs(7,'LTE',short=True)} dondu")

# ============================================================
g6 = group('LTE PRACH config index -> format, FDD (T5.7.1-2)')
_FDD_NA = {30, 46, 60, 61, 62}          # T5.7.1-2'de atanmamis indeksler
for lo, hi, fmt in ((0, 15, 0), (16, 31, 1), (32, 47, 2), (48, 63, 3)):
    for ci in range(lo, hi + 1):
        got = E.get_lte_preamble_format(ci)
        exp = None if ci in _FDD_NA else fmt
        spec(g6, got == exp, f"cfg={ci} -> {exp} olmali, {got} dondu")

g6b = group('LTE PRACH config index -> format, TDD (T5.7.1-4)')
for ci in range(48, 58):
    spec(g6b, E.get_lte_preamble_format(ci, 'TDD') == 4,
         f"TDD cfg={ci} -> format 4 olmali (L_RA=139)")
for ci in range(58, 64):
    spec(g6b, E.get_lte_preamble_format(ci, 'TDD') is None,
         f"TDD cfg={ci} -> N/A olmali")
spec(g6b, E.get_lte_preamble_format(50, 'FDD') == 3,
     "FDD'de ayni indeks format 3 (format 4 FDD'de YOK)")
spec(g6b, E.cell_duplex({'band': 2300}) == 'TDD', "2300 MHz -> TDD")
spec(g6b, E.cell_duplex({'band': 1800}) == 'FDD', "1800 MHz -> FDD")
spec(g6b, E.cell_duplex({'band': 3500}) == 'TDD', "3500 MHz -> TDD")
spec(g6b, E.cell_duplex({'band': 1800, 'duplex': 'TDD'}) == 'TDD',
     "duplex sutunu banttan once gelir")
spec(g6b, E.cell_duplex({}) == 'FDD', "bilgi yoksa FDD varsayilir")

# ============================================================
g7 = group('Cevrimsel kayma penceresi = 1 / delta_f_RA')
# LTE format 0-3: delta_f_RA = 1.25 kHz -> 800 us.  Format 2/3'un 1600/3200 us
# toplam suresi TEKRARDAN gelir, pencereyi degistirmez.
for pcfg, fmt in ((0, 0), (16, 1), (32, 2), (48, 3)):
    info = E.compute_cell_prach_info(
        {'prach_config_index': pcfg, 'zero_correlation_zone': 5}, 'LTE')
    exp_km = round(ref_cell_range_km(26, 839, 1.25), 2)
    spec(g7, abs(info['cell_range_ncs_km'] - exp_km) < 0.02,
         f"LTE cfg={pcfg} (format {fmt}): {info['cell_range_ncs_km']} != {exp_km} km")
# NR long format 3: delta_f_RA = 5 kHz -> 200 us (menzil format 0'in 1/4'u)
spec(g7, abs(ref_cell_range_km(26, 839, 5.0) - 0.93) < 0.02,
     "referans hesap tutarli degil")
# NR short: delta_f_RA = 15 * 2^mu kHz
for scs, exp_km in ((15, ref_cell_range_km(15, 139, 15)),
                    (30, ref_cell_range_km(15, 139, 30))):
    got = E.cell_range_from_ncs(15, 139, ref_window_us(scs))
    spec(g7, abs(got - exp_km) < 0.02,
         f"NR kisa, SCS={scs} kHz: {got:.2f} != {exp_km:.2f} km")
# Motor SCS'i hic bilmiyorsa bu grup gecemez
spec(g7, hasattr(E, 'delta_f_ra_khz') or hasattr(E, '_prach_params') and
     'delta_f_ra_khz' in E._prach_params({'prach_config_index': 30}, 'NR'),
     "_prach_params 'delta_f_ra_khz' dondurmeli")

# ============================================================
g8 = group('Kisitli kumede kok basina preamble (§5.7.2)')
spec(g8, hasattr(E, 'preambles_per_root_restricted'),
     "preambles_per_root_restricted() fonksiyonu yok")
if hasattr(E, 'preambles_per_root_restricted'):
    for u in range(1, NZC, 7):          # tum kok uzayini tara
        for ncs in (15, 32, 76, 119, 279):
            exp = ref_restricted_shifts(u, ncs)
            got = E.preambles_per_root_restricted(u, ncs, NZC)
            spec(g8, got == exp, f"u={u}, Ncs={ncs}: {got} != {exp}")
    spec(g8, E.root_cyclic_shift(129) == 13 and E.root_cyclic_shift(710) == 13,
         "d_u(129) ve d_u(710) = 13 (T5.7.2-4'un ilk iki girdisi)")
    spec(g8, E.get_ncs(8, 'LTE', restricted=True) == 68,
         "kisitli tabloya karsi capraz kontrol")
    # Kisitli kume unrestricted'dan DAIMA daha cok kok ister
    for ncs in (15, 32, 76):
        b, t, w = E.restricted_roots_bounds(ncs)
        unres = E.roots_needed(64, ncs)
        spec(g8, b is not None and b > unres,
             f"Ncs={ncs}: kisitli en iyi ({b}) > unrestricted ({unres}) olmali")
spec(g8, hasattr(E, 'roots_needed_for_cell'),
     "roots_needed_for_cell(rsi, ncs, restricted=...) fonksiyonu yok")
if hasattr(E, 'roots_needed_for_cell'):
    spec(g8, E.roots_needed_for_cell(0, 26, restricted=False) == E.roots_needed(64, 26),
         "unrestricted: baslangic indeksinden bagimsiz")
    spec(g8, E.roots_needed_for_cell(0, 26, restricted=True) is None
         or E.has_root_order(),
         "kok sirasi yuklu degilken kisitli kok sayisi None donmeli "
         "(unrestricted degeri UYDURULMAMALI)")

g8b = group('Kok sirasi tablolari (T5.7.2-4 / T5.7.2-5)')
spec(g8b, E.has_root_order(NZC) and E.has_root_order(139),
     "kok sirasi tablolari yuklenmemis")
from prach_tables import ROOT_ORDER_839, ROOT_ORDER_139
for name, order, nzc in (('T5.7.2-4', ROOT_ORDER_839, 839),
                         ('T5.7.2-5', ROOT_ORDER_139, 139)):
    spec(g8b, len(order) == nzc - 1, f"{name}: {nzc-1} girdi olmali, {len(order)} var")
    spec(g8b, len(set(order)) == len(order), f"{name}: tekrar eden kok var")
    spec(g8b, all(1 <= u < nzc for u in order), f"{name}: aralik disi deger")
    # yapisal degismez: kokler (u, N_ZC-u) ciftleri halinde
    spec(g8b, all(order[i] + order[i+1] == nzc for i in range(0, len(order)-1, 2)),
         f"{name}: cift toplamlari {nzc} degil")
spec(g8b, ROOT_ORDER_839[:6] == (129, 710, 140, 699, 120, 719),
     "T5.7.2-4 ilk 6 girdi spec ile uyusmuyor")
spec(g8b, ROOT_ORDER_139[:6] == (1, 138, 2, 137, 3, 136),
     "T5.7.2-5 ilk 6 girdi spec ile uyusmuyor")
# Kisitli kok sayisi artik KESIN ve baslangic indeksine bagli
_r0 = E.roots_needed_for_cell(0, 68, restricted=True)
_r400 = E.roots_needed_for_cell(400, 68, restricted=True)
spec(g8b, _r0 is not None and _r400 is not None, "kisitli kok sayisi cozulmeli")
spec(g8b, _r0 != _r400,
     f"kisitli kok sayisi baslangica gore degismeli ({_r0} vs {_r400})")
spec(g8b, min(_r0, _r400) > E.roots_needed(64, 68),
     "kisitli kok sayisi unrestricted'dan buyuk olmali")

# ============================================================
g9 = group('PCI / RSI / PRACH config araliklari')
spec(g9, E.pci_count('LTE') == 504 and E.pci_count('NR') == 1008, "PCI sayisi")
spec(g9, E.sss_count('LTE') == 168 and E.sss_count('NR') == 336, "N_ID^(1) sayisi")
spec(g9, E.rsi_count('LTE') == 838 and E.rsi_count('NR', short=True) == 138, "RSI sayisi")
spec(g9, E.prach_config_max('LTE') == 63 and E.prach_config_max('NR') == 255,
     "PRACH config index ust siniri")

# ============================================================
g10 = group('Kok tuketimi: ceil(64 / floor(Nzc/Ncs))')
for ncs, nzc in ((13, 839), (26, 839), (419, 839), (13, 139), (69, 139)):
    spec(g10, E.roots_needed(64, ncs, nzc) == ceil(64 / (nzc // ncs)),
         f"Ncs={ncs}, Nzc={nzc}")
spec(g10, E.roots_needed(64, 0, 839) == 64,
     "Ncs=0 (kayma yok) -> her kok 1 preamble -> 64 kok")

# ============================================================
# NR N_CS tablolari.  Beklenen degerler ETSI TS 138 211 V19.4.0 (= 3GPP TS
# 38.211 v19.4.0) PDF'inden cikarildi ve sayfa goruntusuyle karsilastirildi.
# None = spec'te '-' (yapilandirilamaz) -> motor 0 dondurur.
NR_1P25_UNR = [0, 13, 15, 18, 22, 26, 32, 38, 46, 59, 76, 93, 119, 167, 279, 419]
NR_1P25_A   = [15, 18, 22, 26, 32, 38, 46, 55, 68, 82, 100, 128, 158, 202, 237, None]
NR_1P25_B   = [15, 18, 22, 26, 32, 38, 46, 55, 68, 82, 100, 118, 137, None, None, None]
NR_5_UNR    = [0, 13, 26, 33, 38, 41, 49, 55, 64, 76, 93, 119, 139, 209, 279, 419]
NR_5_A      = [36, 57, 72, 81, 89, 94, 103, 112, 121, 132, 137, 152, 173, 195, 216, 237]
NR_5_B      = [36, 57, 60, 63, 65, 68, 71, 77, 81, 85, 97, 109, 122, 137, None, None]
NR_L139     = [0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 23, 27, 34, 46, 69]

g11 = group('NR Ncs tablolari (T6.3.3.1-5 / -6 / -7)')
for label, exp, kw in (
        ('1.25 kHz sinirsiz', NR_1P25_UNR, dict(delta_f_ra_khz=1.25)),
        ('1.25 kHz tip A',    NR_1P25_A,   dict(delta_f_ra_khz=1.25, restricted='typeA')),
        ('1.25 kHz tip B',    NR_1P25_B,   dict(delta_f_ra_khz=1.25, restricted='typeB')),
        ('5 kHz sinirsiz',    NR_5_UNR,    dict(delta_f_ra_khz=5.0)),
        ('5 kHz tip A',       NR_5_A,      dict(delta_f_ra_khz=5.0, restricted='typeA')),
        ('5 kHz tip B',       NR_5_B,      dict(delta_f_ra_khz=5.0, restricted='typeB')),
        ('L_RA=139',          NR_L139,     dict(short=True))):
    for zcz, e in enumerate(exp):
        got = E.get_ncs(zcz, 'NR', **kw)
        spec(g11, got == (e or 0), f"{label}, zcz={zcz}: {got} != {e}")
# Tablo secimi hucre satirindan da dogru yapiliyor mu: NR format 3 5 kHz
# tablosunu okumali.  Format 3 = pcfg 60-86 (FDD, T6.3.3.2-2) / 40-66 (TDD,
# T6.3.3.2-3).  v3 2026-09'a kadar 1.25 kHz tablosunu okuyordu.
for _row, _lbl in (({'prach_config_index': 60, 'zero_correlation_zone': 5, 'band': 1800}, 'FDD pcfg 60'),
                   ({'prach_config_index': 40, 'zero_correlation_zone': 5, 'band': 3500}, 'TDD pcfg 40')):
    _f3 = E._prach_params(_row, 'NR')
    spec(g11, _f3['delta_f_ra_khz'] == 5.0 and _f3['ncs'] == 41,
         f"NR format 3 ({_lbl}), zcz=5: Ncs={_f3['ncs']} (spec 41), dfRA={_f3['delta_f_ra_khz']}")

# ============================================================
# Huawei'nin Ncs -> Cell Radius tablosu (unrestricted ve high speed sutunlari).
# Ncs=119 -> 15.66 km satiri tablodaki bir yazim hatasi: diger 22 satirin
# deseni 15.95 veriyor, o satir 290 m sapiyor.  Burada kasitli olarak disarida.
HUAWEI_RADIUS_KM = {13: 0.79, 15: 1.08, 18: 1.51, 22: 2.08, 26: 2.65, 32: 3.51,
                    38: 4.37, 46: 5.51, 55: 6.80, 59: 7.37, 68: 8.66, 76: 9.80,
                    82: 10.66, 93: 12.23, 100: 13.23, 128: 17.23, 158: 21.52,
                    167: 22.82, 202: 27.81, 237: 32.82, 279: 38.84, 419: 58.86}
g12 = group('Ncs -> hucre yaricapi, 5.2 us + 2 ornek payi (Sesia 17.10)')
for ncs, km in HUAWEI_RADIUS_KM.items():
    got = E.cell_range_from_ncs(ncs, 839, 800.0)
    spec(g12, abs(got - km) <= 0.02, f"Ncs={ncs}: {got:.3f} km != Huawei {km} km")
    spec(g12, abs(got - ref_cell_range_km(ncs, 839, 1.25)) < 1e-9,
         f"Ncs={ncs}: motor ile bagimsiz referans ayrisiyor")

# ============================================================
g13 = group('cellRange -> zcz: cell_range_from_ncs ile tam ters')
_tables = (('LTE', {}, 0), ('LTE', {'restricted': 'typeA'}, 0),
           ('LTE', {'restricted': 'typeB'}, 0),
           ('NR', {}, 0), ('NR', {}, 60))                 # NR FDD pcfg 60 = format 3
for tech, kw, pcfg in _tables:
    p = E._prach_params({'prach_config_index': pcfg, 'zero_correlation_zone': 1}, tech)
    tbl = E.ncs_table(tech, kw.get('restricted', False), p['is_short'], p['delta_f_ra_khz'])
    for zcz, ncs in sorted(tbl.items()):
        if not ncs:
            continue
        m = E.cell_range_from_ncs(ncs, p['nzc'], p['tseq_us']) * 1000.0
        if m <= 0:
            continue
        z1, n1 = E.derive_zcz_from_cell_range(m, tech, pcfg, **kw)
        spec(g13, n1 == ncs,
             f"{tech} pcfg={pcfg} {kw}: tam esik {m:.1f} m -> Ncs={n1} (beklenen {ncs})")
        bigger = [v for v in tbl.values() if v and v > ncs]
        if bigger:
            z2, n2 = E.derive_zcz_from_cell_range(m + 1.0, tech, pcfg, **kw)
            spec(g13, n2 == min(bigger),
                 f"{tech} pcfg={pcfg} {kw}: esik + 1 m -> Ncs={n2} (beklenen {min(bigger)})")

# Tasma sessiz kalmamali
_z, _n, _ex = E.derive_zcz_from_cell_range(100000, 'LTE', 0, return_exceeded=True)
spec(g13, _ex and _n == 419, f"100 km: Ncs={_n}, exceeded={_ex}")
_z, _n, _ex = E.derive_zcz_from_cell_range(58000, 'LTE', 0, return_exceeded=True)
spec(g13, not _ex and _n == 419, f"58 km: Ncs={_n}, exceeded={_ex}")
spec(g13, E._prach_params({'cell_range': 100000}, 'LTE')['cell_range_exceeded'],
     "_prach_params 100 km hucreyi isaretlemeli")
spec(g13, not E._prach_params({'cell_range': 3000}, 'LTE')['cell_range_exceeded'],
     "_prach_params 3 km hucreyi isaretlememeli")
spec(g13, not E._prach_params({'zero_correlation_zone': 5}, 'LTE')['cell_range_exceeded'],
     "Nokia modunda (cell_range yok) isaret olmamali")

# ============================================================
# NR prach-ConfigurationIndex -> format.  Beklenen araliklar ETSI TS 138 211
# V19.4.0 Tablo 6.3.3.2-2 / -3 / -4'ten (PDF tablolarindan hucre hucre
# cikarildi, ilk sayfalar gozle karsilastirildi).  v3 2026-09'a kadar tek bir
# elle yazilmis kural kullaniyordu: 16-27'yi format 1/2/3, >=28'i kisa sayiyordu.
NR_FMT_FDD = [(0, 27, '0'), (28, 52, '1'), (53, 59, '2'), (60, 86, '3'),
              (87, 107, 'A1'), (108, 116, 'A1/B1'), (117, 136, 'A2'), (137, 146, 'A2/B2'),
              (147, 166, 'A3'), (167, 176, 'A3/B3'), (177, 197, 'B1'), (198, 218, 'B4'),
              (219, 235, 'C0'), (236, 255, 'C2')]
NR_FMT_TDD = [(0, 27, '0'), (28, 33, '1'), (34, 39, '2'), (40, 66, '3'),
              (67, 86, 'A1'), (87, 109, 'A2'), (110, 132, 'A3'), (133, 144, 'B1'),
              (145, 168, 'B4'), (169, 188, 'C0'), (189, 210, 'C2'), (211, 225, 'A1/B1'),
              (226, 240, 'A2/B2'), (241, 255, 'A3/B3'), (256, 262, '0')]
NR_FMT_FR2 = [(0, 28, 'A1'), (29, 58, 'A2'), (59, 88, 'A3'), (89, 111, 'B1'),
              (112, 143, 'B4'), (144, 172, 'C0'), (173, 201, 'C2'), (202, 219, 'A1/B1'),
              (220, 237, 'A2/B2'), (238, 255, 'A3/B3')]
g14 = group('NR prach-ConfigurationIndex -> format (T6.3.3.2-2/-3/-4)')
for label, runs, kw in (('FDD', NR_FMT_FDD, dict(duplex='FDD')),
                        ('TDD', NR_FMT_TDD, dict(duplex='TDD')),
                        ('FR2', NR_FMT_FR2, dict(fr2=True))):
    for a, b, f in runs:
        for i in range(a, b + 1):
            got = E.nr_preamble_format(i, **kw)
            spec(g14, got == f, f"{label} pcfg={i}: {got} != {f}")
    last = runs[-1][1]
    spec(g14, E.nr_preamble_format(last + 1, **kw) is None,
         f"{label} pcfg={last + 1}: tabloda yok, None donmeli")
# Hucre satirindan: bant -> FDD/TDD, FR2 -> hep kisa
for row, exp_short, exp_nzc, lbl in (
        ({'prach_config_index': 20, 'band': 3500}, False, 839, 'TDD pcfg 20 -> format 0'),
        ({'prach_config_index': 30, 'band': 1800}, False, 839, 'FDD pcfg 30 -> format 1'),
        ({'prach_config_index': 30, 'band': 3500}, False, 839, 'TDD pcfg 30 -> format 1'),
        ({'prach_config_index': 70, 'band': 3500}, True, 139, 'TDD pcfg 70 -> A1 (kisa)'),
        ({'prach_config_index': 70, 'band': 1800}, False, 839, 'FDD pcfg 70 -> format 3'),
        ({'prach_config_index': 5, 'band': 28000}, True, 139, 'FR2 pcfg 5 -> A1 (kisa)')):
    p = E._prach_params(row, 'NR')
    spec(g14, p['is_short'] == exp_short and p['nzc'] == exp_nzc,
         f"{lbl}: is_short={p['is_short']}, Nzc={p['nzc']}")
spec(g14, E._prach_params({'prach_config_index': 260, 'band': 1800}, 'NR')['invalid_prach_config'],
     "FDD pcfg 260 tanimsiz, invalid_prach_config isaretlenmeli")

# ============================================================
# Preamble formatinin izin verdigi menzil: gidis-donus hem T_CP'ye hem koruma
# suresine (GT = alt cerceve x 1 ms - T_CP - T_SEQ) sigmali.  TS 36.211 T5.7.1-1.
g15 = group('LTE format menzili = min(T_CP, T_GT) x c / 2')
for fmt, km in ((0, 14.53), (1, 77.34), (2, 29.53), (3, 102.66), (4, 2.19)):
    got = E.cell_range_from_format(fmt)
    spec(g15, abs(got - km) < 0.02, f"format {fmt}: {got:.2f} km != {km} km")
_i = E.compute_cell_prach_info({'cell_range': 38000, 'prach_config_index': 0}, 'LTE')
spec(g15, _i['cell_range_exceeds_format'],
     "38 km hucre format 0'a sigmaz, isaretlenmeli")
_i = E.compute_cell_prach_info({'cell_range': 38000, 'prach_config_index': 16}, 'LTE')
spec(g15, not _i['cell_range_exceeds_format'],
     "38 km hucre format 1'e sigar, isaretlenmemeli")

# ============================================================
print("=" * 74)
print("3GPP TABLO SPESIFIKASYONU")
print("=" * 74)
regressions = 0
pending = 0
for name, r in GROUPS.items():
    total = r['pass'] + r['fail']
    if r['fail'] == 0:
        status = 'GECTI  '
        if r['expect_fail']:
            status = 'ARTIK GECIYOR — expect_fail kaldirilabilir'
    elif r['expect_fail']:
        status = f"BEKLENEN EKSIK ({r['finding']})"
        pending += 1
    else:
        status = 'REGRESYON'
        regressions += 1
    print(f"\n{name}")
    print(f"  {r['pass']}/{total}  {status}")
    for m in r['msgs'][:4]:
        print(f"    - {m}")
    if len(r['msgs']) > 4:
        print(f"    ... +{len(r['msgs']) - 4} tane daha")

print("\n" + "=" * 74)
print(f"Beklenen eksik grup: {pending}   (K-2 / K-3 / K-4 sartnamesi)")
print(f"Regresyon          : {regressions}")
print("=" * 74)
sys.exit(1 if regressions else 0)
