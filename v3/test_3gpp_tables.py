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
    """r = Ncs / (L_RA * delta_f_RA) * c / 2  (gecikme yayilimi payi haric)."""
    return ncs / l_ra * (1.0 / (delta_f_ra_khz * 1000.0)) * 3e8 / 2 / 1000.0


# ============================================================
g1 = group('LTE Ncs — unrestricted (T5.7.2-2)')
EXPECTED_LTE_UNRESTRICTED = [0, 13, 15, 18, 22, 26, 32, 38, 46, 59, 76, 93,
                             119, 167, 279, 419]
for zcz, exp in enumerate(EXPECTED_LTE_UNRESTRICTED):
    spec(g1, E.get_ncs(zcz, 'LTE', restricted=False) == exp,
         f"zcz={zcz}: {E.get_ncs(zcz,'LTE')} != {exp}")

g2 = group('LTE Ncs — restricted set A (T5.7.2-2)')
EXPECTED_LTE_RESTRICTED = [15, 18, 22, 26, 32, 38, 46, 59, 76, 93, 119, 167,
                           279, 419, 839]
for zcz, exp in enumerate(EXPECTED_LTE_RESTRICTED):
    spec(g2, E.get_ncs(zcz, 'LTE', restricted=True) == exp,
         f"zcz={zcz}: {E.get_ncs(zcz,'LTE',restricted=True)} != {exp}")

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
g5 = group('LTE Ncs — format 4 / L_RA=139 (T5.7.2-3)', expect_fail=True, finding='K-3')
EXPECTED_LTE_FMT4 = {0: 2, 1: 4, 2: 6, 3: 8, 4: 10, 5: 12, 6: 15}
for zcz, exp in EXPECTED_LTE_FMT4.items():
    got = E.get_ncs(zcz, 'LTE', short=True)
    spec(g5, got == exp, f"zcz={zcz}: {got} != {exp}")
spec(g5, E.get_ncs(7, 'LTE', short=True) == 0,
     f"zcz=7 format 4'te N/A olmali, {E.get_ncs(7,'LTE',short=True)} dondu")

# ============================================================
g6 = group('LTE PRACH config index -> format, FDD (T5.7.1-2)',
           expect_fail=True, finding='K-3 / Y-5')
for ci in range(0, 16):
    spec(g6, E.get_lte_preamble_format(ci) == 0, f"cfg={ci} -> format 0 olmali")
for ci in range(16, 32):
    spec(g6, E.get_lte_preamble_format(ci) == 1, f"cfg={ci} -> format 1 olmali")
for ci in range(32, 48):
    spec(g6, E.get_lte_preamble_format(ci) == 2, f"cfg={ci} -> format 2 olmali")
for ci in range(48, 64):
    got = E.get_lte_preamble_format(ci)
    spec(g6, got == 3, f"cfg={ci} -> format 3 olmali (FDD'de format 4 YOK), {got} dondu")

# ============================================================
g7 = group('Cevrimsel kayma penceresi = 1 / delta_f_RA',
           expect_fail=True, finding='K-4')
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

g8b = group('Kisitli kume — tam cozum icin kok sirasi tablosu',
            expect_fail=True, finding='K-2 / tablo bekleniyor')
spec(g8b, E.has_root_order(NZC),
     "TS 36.211 T5.7.2-4 (mantiksal->fiziksel kok sirasi) yuklenmemis; "
     "kisitli kume kok sayisi ancak medyan tahminle veriliyor")

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
