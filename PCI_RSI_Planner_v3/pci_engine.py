"""
3GPP PCI/RSI Planning Engine v2.0
==================================
RSI Planning: Cell-range-aware using PRACH Config Index & zeroCorrelationZoneConfig
  - 3GPP TS 36.211 Table 5.7.1-2: PRACH Config Index → Preamble Format
  - 3GPP TS 36.211 Table 5.7.2-2/3: zeroCorrelationZoneConfig → Ncs
  - Cell Range = (Ncs / Nzc) * Tseq * c / 2
  - Roots needed = ceil(64 / floor(Nzc / Ncs))
  - RSI overlap when neighbor cells consume overlapping root sequence indices
"""

import numpy as np
import pandas as pd
import random
import time
import math as _math
from math import radians, sin, cos, sqrt, atan2, degrees, floor, ceil
from collections import defaultdict
from typing import Dict, Tuple, Set

# ============================================================
# Constants
# ============================================================
LTE_PCI_COUNT = 504
NR_PCI_COUNT = 1008
LTE_RSI_COUNT = 838
NR_RSI_LONG_COUNT = 838
NR_RSI_SHORT_COUNT = 138
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT = 3e8
NZC_LONG = 839
NZC_SHORT = 139

# Cyclic-shift dimensioning margin.  A shift window of N_CS samples has to hold
# the round-trip delay PLUS the channel delay spread PLUS a few samples of
# receiver pulse-shaping spill-over:
#
#     N_CS >= ceil((20/3 * r + tau_ds) * N_ZC / T_SEQ) + n_g       r in km, us
#
# Sesia/Toufik/Baker, "LTE - The UMTS Long Term Evolution", 2nd ed., eq. 17.10
# and p.390: the 3GPP N_CS set was designed "assuming a delay spread of 5.2 us
# and 2 guard samples n_g".  (The book prints "- tau_ds"; with that sign no row
# of any vendor table reproduces, with "+" every row does.)
#
# Verified against the Huawei Ncs -> cell radius table: 22 of 23 rows within
# 15 m (the 23rd, Ncs=119 -> 15.66 km, is a typo in that table; the pattern
# gives 15.95).  Without this margin v3 reported every range ~1.07 km too long
# and, in Huawei mode, picked an Ncs one step too small for many radii.
#
# Applied only where it is verified: L_RA=839 with delta_f_RA=1.25 kHz (LTE
# formats 0-3, NR formats 0-2).  For NR format 3 and the L_RA=139 formats the
# same 5.2 us would eat most of the window (NR short @30 kHz, Ncs<=15 would
# cover no cell at all), and no vendor table was available to confirm what is
# assumed there — so those keep the plain round-trip formula.
PRACH_DELAY_SPREAD_US = 5.2
PRACH_GUARD_SAMPLES = 2

# ============================================================
# Technology & PCI range — single source of truth
# ============================================================
# The **UI selection is the sole authority** for technology.  The cell data
# is never consulted: whatever the operator picks in the sidebar decides the
# PCI range, the mod-N rule set and the Ncs tables for the whole run.
#
# 3GPP TS 36.211 §6.11    LTE: N_ID^cell = 3·N_ID^(1) + N_ID^(2),
#                              N_ID^(1) ∈ 0..167, N_ID^(2) ∈ 0..2  →  0-503
# 3GPP TS 38.211 §7.4.2.1 NR : N_ID^cell = 3·N_ID^(1) + N_ID^(2),
#                              N_ID^(1) ∈ 0..335, N_ID^(2) ∈ 0..2  →  0-1007

def norm_tech(technology):
    """Normalise any technology label to exactly 'NR' or 'LTE'.

    Accepts 'NR', 'nr', '5G', 'NR (5G)', … — everything else is LTE.
    Guards against the silent-LTE-fallback that a bare ``technology == 'NR'``
    comparison produces for e.g. 'nr' or '5G'.
    """
    t = str(technology).strip().upper()
    return 'NR' if ('NR' in t or '5G' in t) else 'LTE'


def pci_count(technology='LTE'):
    """Number of distinct PCIs: 504 (LTE) / 1008 (NR)."""
    return NR_PCI_COUNT if norm_tech(technology) == 'NR' else LTE_PCI_COUNT


def pci_max(technology='LTE'):
    """Highest valid PCI value: 503 (LTE) / 1007 (NR)."""
    return pci_count(technology) - 1


def sss_count(technology='LTE'):
    """Number of cell-identity groups N_ID^(1): 168 (LTE) / 336 (NR)."""
    return pci_count(technology) // 3


def rsi_count(technology='LTE', short=False):
    """Number of PRACH root sequence indices: 838 (L=839) / 138 (L=139)."""
    return NR_RSI_SHORT_COUNT if short else (
        NR_RSI_LONG_COUNT if norm_tech(technology) == 'NR' else LTE_RSI_COUNT)


# ============================================================
# Carrier (frequency layer) identity — K-1
# ============================================================
# Collision, confusion, mod-N and RSI conflicts are only defined between cells
# on the SAME carrier: a UE measurement report carries (PCI, ARFCN), so two
# cells on different frequencies can never be confused for one another.
# This scoping is physics, not strategy — it is always on.  What IS a strategy
# choice is the planning granularity (one PCI per physical sector across all
# its carriers, vs one PCI per sector per carrier); see PLANNING_SCOPES.

PLANNING_SCOPES = ('sector', 'carrier')

CARRIER_UNKNOWN = '?'


def cell_carrier(row):
    """Carrier key for one cell row.

    Priority: explicit earfcn/arfcn > band column > band parsed from the cell
    ID naming convention.  Returns CARRIER_UNKNOWN when nothing identifies the
    carrier.

    An explicit ARFCN is the only fully reliable source: a band alone cannot
    separate two carriers in the same band (e.g. an 1800 MHz 20 MHz carrier
    and an 1800 MHz 10 MHz carrier), and the cell-ID prefix cannot either —
    in real data one prefix is used for two different bands.
    """
    for key, tag in (('earfcn', 'AR'), ('arfcn', 'AR')):
        v = row.get(key)
        if v is not None:
            try:
                if not pd.isna(v) and float(v) > 0:
                    return f'{tag}{int(float(v))}'
            except (TypeError, ValueError):
                pass
    v = row.get('band')
    if v is not None:
        try:
            if not pd.isna(v) and float(v) > 0:
                return f'B{int(float(v))}'
        except (TypeError, ValueError):
            s = str(v).strip()
            if s and s.lower() != 'nan':
                return f'B{s}'
    v = row.get('band_mhz')
    if v is not None:
        try:
            if not pd.isna(v) and float(v) > 0:
                return f'B{int(float(v))}'
        except (TypeError, ValueError):
            pass
    return CARRIER_UNKNOWN


def build_carrier_map(df):
    """cell_id -> carrier key.  See cell_carrier() for the resolution order.

    A pre-computed 'carrier' column is used when present, but an EMPTY cell in
    it falls back to resolving the row rather than becoming unknown.  That case
    is not hypothetical: the new-cell finder concatenates rows onto a frame that
    already carries the column, so every new cell arrives with it blank — and a
    cell whose carrier is unknown shares a carrier with nothing, so it would be
    planned against no constraints at all and silently get whatever PCI came
    first.
    """
    has_col = 'carrier' in df.columns
    out = {}
    for _, r in df.iterrows():
        cid = str(r['cell_id'])
        val = r.get('carrier') if has_col else None
        if val is not None and pd.notna(val) and str(val).strip():
            out[cid] = str(val)
        else:
            out[cid] = cell_carrier(r)
    return out


def enrich_carrier_column(df):
    """Add a 'carrier' column derived from earfcn/arfcn, band, or the cell ID."""
    df = df.copy()
    df['carrier'] = [cell_carrier(r) for _, r in df.iterrows()]
    return df


def carrier_report(carrier_map):
    """(n_carriers, n_unknown, {carrier: cell_count}) for messaging."""
    counts = defaultdict(int)
    for c in carrier_map.values():
        counts[c] += 1
    known = {k: v for k, v in counts.items() if k != CARRIER_UNKNOWN}
    return len(known), counts.get(CARRIER_UNKNOWN, 0), dict(counts)


def same_carrier(carrier_map, a, b):
    """True if two cells share a carrier.

    With no carrier information at all every cell resolves to CARRIER_UNKNOWN,
    so the whole network behaves as a single carrier — identical to the
    pre-K-1 behaviour.  Cells whose carrier could not be resolved are kept in
    their own bucket rather than being made to conflict with everything.
    """
    if not carrier_map:
        return True
    return carrier_map.get(str(a), CARRIER_UNKNOWN) == carrier_map.get(str(b), CARRIER_UNKNOWN)


def scope_neighbors_by_carrier(neighbors, carrier_map):
    """Drop cross-carrier edges from a neighbour graph.

    Used for collision / mod-N / RSI checks.  NOT used for confusion: a cell
    on one carrier can legitimately have inter-frequency neighbours on another,
    and two of those neighbours sharing a PCI is a real confusion on THEIR
    carrier — so confusion traverses the full graph and only requires the
    ambiguous pair to share a carrier.
    """
    if not carrier_map:
        return neighbors
    out = defaultdict(set)
    for c, nbs in neighbors.items():
        cc = carrier_map.get(str(c), CARRIER_UNKNOWN)
        for nb in nbs:
            if carrier_map.get(str(nb), CARRIER_UNKNOWN) == cc:
                out[str(c)].add(str(nb))
    return out


def split_sector_groups_by_carrier(sector_groups, cell_to_sector, carrier_map):
    """Split each sector group so every carrier gets its own PCI decision.

    'sector' planning scope keeps the groups as they are: one PCI for the whole
    physical sector across all of its carriers.  'carrier' scope calls this, so
    a sector's 3500 MHz cells and its 1800 MHz cells are planned independently.
    """
    new_groups = {}
    new_c2s = {}
    for sec_key, members in sector_groups.items():
        by_car = defaultdict(list)
        for m in members:
            by_car[carrier_map.get(str(m), CARRIER_UNKNOWN)].append(m)
        for car, cells in by_car.items():
            key = f'{sec_key}@{car}'
            new_groups[key] = cells
            for m in cells:
                new_c2s[str(m)] = key
    return new_groups, new_c2s


def prach_config_max(technology='LTE'):
    """Highest valid PRACH configuration index for the technology.

    LTE : prach-ConfigIndex            INTEGER (0..63)   TS 36.331 PRACH-ConfigInfo
    NR  : prach-ConfigurationIndex     INTEGER (0..255)  TS 38.331 RACH-ConfigGeneric
    """
    return 255 if norm_tech(technology) == 'NR' else 63


ATTEMPT_W_MIN = 0.25    # a relation with no measured traffic still matters a little
ATTEMPT_W_MAX = 4.00    # cap so one very hot pair cannot dominate the objective


def build_attempt_weights(nbr_attempts, w_min=ATTEMPT_W_MIN, w_max=ATTEMPT_W_MAX):
    """Map each neighbour pair to an optimisation weight from its HO attempts.

    A conflict on a relation carrying 38,000 handovers a day is not the same
    problem as a conflict on a relation nobody uses, and when PCIs run short
    the planner has to push the unavoidable conflicts onto the quiet pairs.
    Without weights it picks arbitrarily.

    Raw counts are unusable directly - they span 0 to >150,000 here, so a
    linear weight would let a handful of pairs drown out everything else and
    the annealer would stop making progress on the rest.  log1p normalised
    against the 99th percentile keeps the ordering while compressing the range
    to w_min..w_max (16x between a dead pair and a hot one).

    Returns (weight_lookup, stats).  weight_lookup(a, b) -> float.
    """
    if not nbr_attempts:
        return (lambda a, b: 1.0), {'enabled': False}
    vals = sorted(v for v in nbr_attempts.values() if v and v > 0)
    if not vals:
        return (lambda a, b: 1.0), {'enabled': False}
    ref = vals[min(int(len(vals) * 0.99), len(vals) - 1)]
    denom = _math.log1p(ref) or 1.0
    span = w_max - w_min
    cache = {}
    for pair, att in nbr_attempts.items():
        a = att if att and att > 0 else 0
        w = w_min + span * min(_math.log1p(a) / denom, 1.0)
        cache[(str(pair[0]), str(pair[1]))] = w

    def lookup(a, b):
        a, b = str(a), str(b)
        key = (a, b) if a < b else (b, a)
        return cache.get(key, w_min)

    stats = {'enabled': True, 'pairs': len(cache), 'p99_attempts': ref,
             'median_attempts': vals[len(vals) // 2],
             'max_attempts': vals[-1], 'w_min': w_min, 'w_max': w_max}
    return lookup, stats


def conflicted_attempts(conflict_tables, nbr_attempts):
    """Total HO attempts riding on relations affected by a conflict.

    The health score counts conflicts; this counts the traffic exposed to them,
    which is what actually shows up as dropped or failed handovers.  Reported
    alongside the score, not folded into it.

    Collision / mod-N / RSI rows name two cells that ARE neighbours, so the
    affected relation is the pair itself.  A confusion row is different: cell_1
    and cell_2 are the two ambiguous cells and are frequently not neighbours of
    each other — the traffic at risk rides on the two relations from the common
    neighbour, so those are the pairs that count.
    """
    if not nbr_attempts:
        return None
    total = 0
    seen = set()

    def _add(a, b):
        nonlocal total
        a, b = str(a), str(b)
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        total += nbr_attempts.get(key, 0)

    for tbl in conflict_tables:
        if tbl is None or len(tbl) == 0:
            continue
        if 'common_neighbor' in tbl.columns:
            for ca, c1, c2 in zip(tbl['common_neighbor'].astype(str),
                                  tbl['cell_1'].astype(str),
                                  tbl['cell_2'].astype(str)):
                _add(ca, c1)
                _add(ca, c2)
        else:
            for a, b in zip(tbl['cell_1'].astype(str), tbl['cell_2'].astype(str)):
                _add(a, b)
    return total


def pci_in_range(pci, technology='LTE'):
    """True if `pci` is a valid PCI for the technology."""
    try:
        p = int(pci)
    except (TypeError, ValueError):
        return False
    return 0 <= p <= pci_max(technology)


def assert_pci_range(values, technology, where):
    """Raise ValueError if any value falls outside the technology's PCI range.

    Every planner calls this on its own output, so an out-of-range PCI can
    never reach the user — not even when the input data carries PCIs from a
    different technology.  Non-numeric placeholders ('—') are ignored.
    """
    bad = []
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(v, str) and not v.strip().lstrip('-').isdigit():
            continue  # placeholder such as '—'
        if not pci_in_range(v, technology):
            bad.append(v)
    if bad:
        raise ValueError(
            f"{where}: {norm_tech(technology)} icin gecersiz PCI uretildi "
            f"(izinli aralik 0-{pci_max(technology)}): {sorted(set(bad))[:10]}")

# ============================================================
# 3GPP TS 36.211 Table 5.7.2-2 – Ncs UNRESTRICTED (format 0-3, Nzc=839)
# ============================================================
LTE_NCS_UNRESTRICTED = {
    0:0, 1:13, 2:15, 3:18, 4:22, 5:26, 6:32, 7:38,
    8:46, 9:59, 10:76, 11:93, 12:119, 13:167, 14:279, 15:419}

# Tables below come from prach_tables.py, extracted from the spec text and
# structurally validated — not typed in from memory.  The v2 restricted-set
# table was the unrestricted column shifted by one from zcz=7 up (59, 76, 93,
# 119, 167, 279, 419, 839 instead of 55, 68, 82, 100, 128, 158, 202, 237),
# and the test suite had the same wrong values, so it passed.
from prach_tables import (
    LTE_PRACH_FORMAT_FDD, LTE_PRACH_FORMAT_TDD,
    LTE_NCS_RESTRICTED_A_SPEC, LTE_NCS_RESTRICTED_B_SPEC,
    LTE_NCS_FORMAT4_SPEC, ROOT_ORDER_839, ROOT_ORDER_139,
    NR_NCS_1P25_UNRESTRICTED_SPEC, NR_NCS_1P25_RESTRICTED_A_SPEC,
    NR_NCS_1P25_RESTRICTED_B_SPEC, NR_NCS_5_UNRESTRICTED_SPEC,
    NR_NCS_5_RESTRICTED_A_SPEC, NR_NCS_5_RESTRICTED_B_SPEC, NR_NCS_L139_SPEC,
    NR_PRACH_FORMAT_FR1_PAIRED, NR_PRACH_FORMAT_FR1_UNPAIRED, NR_PRACH_FORMAT_FR2,
)

def _clean(d):
    """Drop N/A entries so a lookup miss returns 0 (= not configurable)."""
    return {k: v for k, v in d.items() if v is not None}

# 3GPP TS 36.211 Table 5.7.2-2 – Ncs RESTRICTED set, formats 0-3
LTE_NCS_RESTRICTED = _clean(LTE_NCS_RESTRICTED_A_SPEC)      # type A
LTE_NCS_RESTRICTED_B = _clean(LTE_NCS_RESTRICTED_B_SPEC)    # type B (Rel-13+)

# 3GPP TS 36.211 Table 5.7.2-3 – Ncs for PREAMBLE FORMAT 4 (TDD only, L_RA=139)
LTE_NCS_FORMAT4 = _clean(LTE_NCS_FORMAT4_SPEC)

# 3GPP TS 36.211 Table 5.7.1-1 – Preamble Format params
LTE_PREAMBLE_FORMATS = {
    0: {'tcp_us':103.13,  'tseq_us':800.0,  'subframes':1},
    1: {'tcp_us':684.38,  'tseq_us':800.0,  'subframes':2},
    2: {'tcp_us':203.13,  'tseq_us':1600.0, 'subframes':2},
    3: {'tcp_us':684.38,  'tseq_us':1600.0, 'subframes':3},
    4: {'tcp_us':14.58,   'tseq_us':133.33, 'subframes':1}}

# 3GPP TS 38.211 Table 6.3.3.1-5 – NR Ncs, L_RA=839, delta_f_RA=1.25 kHz
# (formats 0, 1, 2).  Same values as LTE Table 5.7.2-2.
NR_NCS_LONG = _clean(NR_NCS_1P25_UNRESTRICTED_SPEC)
NR_NCS_LONG_RESTRICTED_A = _clean(NR_NCS_1P25_RESTRICTED_A_SPEC)
NR_NCS_LONG_RESTRICTED_B = _clean(NR_NCS_1P25_RESTRICTED_B_SPEC)

# 3GPP TS 38.211 Table 6.3.3.1-6 – NR Ncs, L_RA=839, delta_f_RA=5 kHz (format 3).
# A table of its own: v3 used to read format 3 from the 1.25 kHz table, so
# zcz=5 came out as Ncs=26 where the standard says 41.
NR_NCS_5KHZ = _clean(NR_NCS_5_UNRESTRICTED_SPEC)
NR_NCS_5KHZ_RESTRICTED_A = _clean(NR_NCS_5_RESTRICTED_A_SPEC)
NR_NCS_5KHZ_RESTRICTED_B = _clean(NR_NCS_5_RESTRICTED_B_SPEC)

# 3GPP TS 38.211 Table 6.3.3.1-7 – NR Ncs, L_RA=139 (short formats A/B/C).
NR_NCS_SHORT = _clean(NR_NCS_L139_SPEC)

# NR PRACH Preamble Format mapping
# Long sequence (L=839): Formats 0, 1, 2, 3
# Short sequence (L=139): Formats A1, A2, A3, B1, B2, B3, B4, C0, C2
# 3GPP TS 38.211 §6.3.3.1
NR_PREAMBLE_FORMATS = {
    0: {'nzc': NZC_LONG,  'tseq_us': 800.0,  'label': 'Format 0 (Long)'},
    1: {'nzc': NZC_LONG,  'tseq_us': 800.0,  'label': 'Format 1 (Long)'},
    2: {'nzc': NZC_LONG,  'tseq_us': 1600.0, 'label': 'Format 2 (Long)'},
    3: {'nzc': NZC_LONG,  'tseq_us': 1600.0, 'label': 'Format 3 (Long)'},
    # Short formats — prach_config_index typically 0-27 maps to long (0-3),
    # values ≥ 28 map to short (A1-C2). Simplified mapping:
    'A1': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format A1 (Short)'},
    'A2': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format A2 (Short)'},
    'A3': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format A3 (Short)'},
    'B1': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format B1 (Short)'},
    'B4': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format B4 (Short)'},
    'C0': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format C0 (Short)'},
    'C2': {'nzc': NZC_SHORT, 'tseq_us': 133.33, 'label': 'Format C2 (Short)'},
}

# ============================================================
# PRACH subcarrier spacing (delta_f_RA) — K-4
# ============================================================
# The cyclic-shift window a cell gets from Ncs is Ncs / (L_RA * delta_f_RA),
# i.e. it is set by the duration of ONE Zadoff-Chu sequence = 1 / delta_f_RA.
# A format that repeats the sequence (LTE format 2/3, NR format 1/2/3) has a
# longer TOTAL T_SEQ, but the repetition buys coverage through energy, not
# through a wider shift window.  v2 used the total, so LTE format 2 and 3 came
# out with twice the cell range they actually have.
#
# LTE formats 0-3  : 1.25 kHz  -> 800 us per sequence   (TS 36.211 T5.7.1-1)
# LTE format 4     : 7.5 kHz   -> 133.33 us
# NR formats 0,1,2 : 1.25 kHz  -> 800 us                (TS 38.211 T6.3.3.1-1)
# NR format 3      : 5 kHz     -> 200 us   <- a quarter of format 0
# NR short A/B/C   : 15*2^mu kHz -> 66.67 / 33.33 / 16.67 / 8.33 us

LTE_DELTA_F_RA_KHZ = {0: 1.25, 1: 1.25, 2: 1.25, 3: 1.25, 4: 7.5}

# NR prach-ConfigurationIndex -> preamble format comes from three different
# tables (TS 38.211 T6.3.3.2-2 FDD, -3 TDD, -4 FR2), now in prach_tables.py.
# Until 2026-09 v3 used one hand-written rule for all three — 0-15 format 0,
# 16-19 format 1, 20-22 format 2, 23-27 format 3, >=28 short — which the spec
# contradicts almost everywhere: 0-27 are ALL format 0 in both FR1 tables,
# 28-86 (FDD) / 28-66 (TDD) are long formats 1-3, and FR2 is short throughout.


def sequence_window_us(delta_f_ra_khz):
    """Duration of one ZC sequence = 1 / delta_f_RA, in microseconds."""
    return 1000.0 / float(delta_f_ra_khz)


def nr_short_delta_f_khz(scs_khz=None, band_mhz=None):
    """delta_f_RA for an NR short preamble: 15 * 2^mu kHz.

    Comes from msg1-SubcarrierSpacing when the data carries it.  Otherwise it
    is defaulted from the band, because getting this wrong scales the cell
    range by 2-8x: n78 runs 30 kHz, sub-3 GHz NR runs 15 kHz, FR2 runs 120 kHz.
    """
    if scs_khz is not None:
        try:
            if not pd.isna(scs_khz) and float(scs_khz) > 0:
                return float(scs_khz)
        except (TypeError, ValueError):
            pass
    try:
        b = float(band_mhz) if band_mhz is not None and not pd.isna(band_mhz) else None
    except (TypeError, ValueError):
        b = None
    if b is None:
        return 30.0          # most common NR deployment
    if b >= 24000:
        return 120.0         # FR2
    if b >= 3000:
        return 30.0          # n77 / n78
    return 15.0              # sub-3 GHz NR


def nr_is_fr2(band_mhz=None, scs_khz=None):
    """FR2 if the band is above 24 GHz, or msg1-SubcarrierSpacing is 60/120 kHz
    (those spacings exist for PRACH only in FR2)."""
    for v, lim in ((band_mhz, 24000.0), (scs_khz, 60.0)):
        try:
            if v is not None and not pd.isna(v) and float(v) >= lim:
                return True
        except (TypeError, ValueError):
            pass
    return False


def nr_preamble_format(prach_config_index, duplex='FDD', fr2=False):
    """Preamble format string for an NR prach-ConfigurationIndex, or None when
    the index does not exist in the table that applies (e.g. 256-262 in FDD)."""
    try:
        pcfg = int(prach_config_index)
    except (TypeError, ValueError):
        return None
    if fr2:
        table = NR_PRACH_FORMAT_FR2
    elif str(duplex).upper() == 'TDD':
        table = NR_PRACH_FORMAT_FR1_UNPAIRED
    else:
        table = NR_PRACH_FORMAT_FR1_PAIRED
    return table[pcfg] if 0 <= pcfg < len(table) else None


def get_nr_preamble_info(prach_config_index=0, duplex='FDD', fr2=False):
    """NR preamble sequence length and subcarrier spacing from the config index.

    The format comes from the table that applies to the cell: FR1 FDD (T6.3.3.2-2),
    FR1 TDD (T6.3.3.2-3) or FR2 (T6.3.3.2-4).  Long formats 0-2 run at 1.25 kHz,
    format 3 at 5 kHz; short formats take delta_f_RA from msg1-SubcarrierSpacing,
    so None is returned for them and the caller fills it in.

    An index that the applicable table does not define falls back to format 0;
    callers flag it through nr_preamble_format() returning None.

    Returns (is_short, nzc, delta_f_ra_khz, format_label).
    """
    try:
        pcfg = int(prach_config_index) if not pd.isna(prach_config_index) else 0
    except (TypeError, ValueError):
        pcfg = 0
    fmt = nr_preamble_format(pcfg, duplex, fr2) or '0'
    if fmt in ('0', '1', '2', '3'):
        # Format 3 is the odd one out: 5 kHz, so a quarter of the shift window
        # that formats 0-2 get from the same Ncs, and its own Ncs table.
        return False, NZC_LONG, (5.0 if fmt == '3' else 1.25), f'Format {fmt} (Long)'
    return True, NZC_SHORT, None, f'Format {fmt} (Short)'

# ============================================================
# Band / Bandwidth Detection from Cell ID Naming Convention
# ============================================================
def parse_band_info(cell_id: str) -> dict:
    """Detect band and bandwidth from cell_id prefix/suffix convention.

    Rules (Turkish operator naming):
      L*       → 800 MHz  (LTE Band 20)
      T*A/D/G/J/M/P  → 1800 MHz, 20 MHz BW
      T*B/E/H/K/N/R  → 1800 MHz, 10 MHz BW
      T*       → 1800 MHz (generic)
      Z*       → 2100 MHz (LTE Band 1)
      E*       → 2600 MHz (LTE Band 7)
      K*       → 700 MHz  (LTE Band 28)
      G*       → 900 MHz  (LTE Band 8)
      N*       → 3500 MHz (NR n78)
      F*       → 2300 MHz (LTE Band 40 / NR n40)
      H*       → 3700 MHz (NR n77)
    Returns dict with keys: band_mhz, bandwidth_mhz, band_label
    """
    cid = str(cell_id).strip().upper()
    if not cid:
        return {'band_mhz': None, 'bandwidth_mhz': None, 'band_label': 'Bilinmeyen'}

    first = cid[0]
    last = cid[-1] if len(cid) > 1 else ''

    if first == 'K':
        return {'band_mhz': 700, 'bandwidth_mhz': None, 'band_label': '700 MHz'}
    elif first == 'L':
        return {'band_mhz': 800, 'bandwidth_mhz': None, 'band_label': '800 MHz'}
    elif first == 'G':
        return {'band_mhz': 900, 'bandwidth_mhz': None, 'band_label': '900 MHz'}
    elif first == 'T':
        if last in ('A', 'D', 'G', 'J', 'M', 'P'):
            return {'band_mhz': 1800, 'bandwidth_mhz': 20, 'band_label': '1800 MHz / 20 MHz'}
        elif last in ('B', 'E', 'H', 'K', 'N', 'R'):
            return {'band_mhz': 1800, 'bandwidth_mhz': 10, 'band_label': '1800 MHz / 10 MHz'}
        else:
            return {'band_mhz': 1800, 'bandwidth_mhz': None, 'band_label': '1800 MHz'}
    elif first == 'Z':
        return {'band_mhz': 2100, 'bandwidth_mhz': None, 'band_label': '2100 MHz'}
    elif first == 'F':
        return {'band_mhz': 2300, 'bandwidth_mhz': None, 'band_label': '2300 MHz'}
    elif first == 'E':
        return {'band_mhz': 2600, 'bandwidth_mhz': None, 'band_label': '2600 MHz'}
    elif first == 'N':
        return {'band_mhz': 3500, 'bandwidth_mhz': None, 'band_label': '3500 MHz'}
    elif first == 'H':
        return {'band_mhz': 3700, 'bandwidth_mhz': None, 'band_label': '3700 MHz'}
    else:
        return {'band_mhz': None, 'bandwidth_mhz': None, 'band_label': 'Bilinmeyen'}


def enrich_band_columns(df):
    """Add band_mhz, bandwidth_mhz, band_label columns to DataFrame based on cell_id."""
    bands = df['cell_id'].apply(parse_band_info)
    df = df.copy()
    df['band_mhz'] = bands.apply(lambda x: x['band_mhz'])
    df['bandwidth_mhz'] = bands.apply(lambda x: x['bandwidth_mhz'])
    df['band_label'] = bands.apply(lambda x: x['band_label'])
    return df


# ============================================================
# Cell ID → Site Name & Sector Number Extraction
# ============================================================
def _extract_site_name(cell_id: str):
    """Extract base site name from a Turkish operator cell ID.

    Naming convention:
        [Band prefix (1 char)][Base site name][Sector/carrier letter (1 char)]
        Examples: EAS0425G  → site='AS0425'
                  TAS0425H  → site='AS0425'
                  LAS0403G  → site='AS0403'
                  ZAS0403G  → site='AS0403'
        Band prefixes: E(2600), T(1800), L(800), Z(2100), K(700), G(900), etc.
        Last letter determines SECTOR and CARRIER:
          A,B,C → Sector 1 (carrier 1,2,3)
          D,E,F → Sector 2 (carrier 1,2,3)
          G,H,I → Sector 3 (carrier 1,2,3)
          J,K,L → Sector 4 (carrier 1,2,3)  — typically indoor
          M,N,O → Sector 5 (carrier 1,2,3)  — typically indoor
          etc.

    Returns base_site (str) or None if pattern doesn't match.
    """
    cid = str(cell_id).strip()
    if len(cid) < 3:
        return None
    last = cid[-1].upper()
    if not last.isalpha():
        return None
    first = cid[0].upper()
    if not first.isalpha():
        return None
    # Base site = everything between first and last character
    base_site = cid[1:-1]
    if not base_site:
        return None
    return base_site.upper()


def _extract_sector_number(cell_id: str):
    """Extract sector number from the last letter of a Turkish operator cell ID.

    Sector mapping (last letter → sector number):
        A,B,C → 1    D,E,F → 2    G,H,I → 3
        J,K,L → 4    M,N,O → 5    P,Q,R → 6
        S,T,U → 7    V,W,X → 8    Y,Z   → 9

    Cells with the same base site name AND same sector number are **co-sector**
    (same physical antenna, different band/carrier).  They share the same PCI.

    Returns sector_number (int 1-9) or None if pattern doesn't match.
    """
    cid = str(cell_id).strip()
    if len(cid) < 3:
        return None
    last = cid[-1].upper()
    if not last.isalpha():
        return None
    first = cid[0].upper()
    if not first.isalpha():
        return None
    return (ord(last) - ord('A')) // 3 + 1


def _is_same_site_by_id(cell_id_1: str, cell_id_2: str) -> bool:
    """Check if two cells belong to the same physical site based on cell ID.
    Same site = same base site name (middle part of cell ID).
    NOTE: This does NOT mean co-sector! Co-sector requires same sector number.
    """
    s1 = _extract_site_name(cell_id_1)
    s2 = _extract_site_name(cell_id_2)
    if s1 is None or s2 is None:
        return False
    return s1 == s2


# Module-level set of site names where naming convention is unreliable
# (unbalanced band counts across sectors).  Populated by detect_sector_groups().
_MIXED_INDOOR_SITES: set = set()

# Module-level set of cell IDs detected as indoor.  Populated by detect_sector_groups().
_DETECTED_INDOOR_CELLS: set = set()


def _extract_band_prefix(cell_id: str):
    """Extract band prefix (first character) from a Turkish operator cell ID.
    Band prefixes: E(2600), T(1800), L(800), Z(2100), K(700), G(900), etc.
    Returns uppercase letter or None.
    """
    cid = str(cell_id).strip()
    if len(cid) < 3:
        return None
    first = cid[0].upper()
    if not first.isalpha():
        return None
    return first


def _is_indoor_cell(cell_id: str) -> bool:
    """Detect if a cell is an indoor cell.

    Uses _DETECTED_INDOOR_CELLS set populated by detect_sector_groups().
    Fallback heuristic (before detect_sector_groups runs): sector ≥ 4 + T-band.
    """
    cid = str(cell_id).strip()
    if cid in _DETECTED_INDOOR_CELLS:
        return True
    # Fallback heuristic when detect_sector_groups hasn't run yet
    if not _DETECTED_INDOOR_CELLS:
        sec = _extract_sector_number(cid)
        bp = _extract_band_prefix(cid)
        if sec is not None and bp is not None and sec >= 4 and bp == 'T':
            return True
    return False


def get_sector_label(cell_id: str) -> str:
    """Return a human-readable sector label such as 'O-1', 'I-4'.

    Indoor sectors (sector ≥4, T-band) → 'I-<n>'
    Outdoor sectors → 'O-<n>'
    If sector number cannot be determined → '—'
    """
    sec = _extract_sector_number(cell_id)
    if sec is None:
        return '—'
    if _is_indoor_cell(cell_id):
        return f'I-{sec}'
    return f'O-{sec}'


def get_cell_environment(cell_id: str) -> str:
    """Return 'Indoor' or 'Outdoor' based on naming convention."""
    if _is_indoor_cell(cell_id):
        return 'Indoor'
    return 'Outdoor'


def enrich_df_with_sector_info(df_in):
    """Add 'sektör' and 'ortam' columns to a DataFrame.

    Works with DataFrames that have:
      - 'cell_id' column  → adds sektör, ortam
      - 'cell_1' / 'cell_2' columns (conflict tables) → adds sektör_1, ortam_1, sektör_2, ortam_2

    Skips columns that already exist.  Returns new DataFrame (does not modify input).
    """
    if df_in is None or len(df_in) == 0:
        return df_in
    df = df_in.copy()
    if 'cell_id' in df.columns and 'sektör' not in df.columns:
        df.insert(1, 'sektör', df['cell_id'].astype(str).map(get_sector_label))
        df.insert(2, 'ortam', df['cell_id'].astype(str).map(get_cell_environment))
    if 'cell_1' in df.columns and 'sektör_1' not in df.columns:
        idx = df.columns.get_loc('cell_1') + 1
        df.insert(idx, 'sektör_1', df['cell_1'].astype(str).map(get_sector_label))
        df.insert(idx + 1, 'ortam_1', df['cell_1'].astype(str).map(get_cell_environment))
    if 'cell_2' in df.columns and 'sektör_2' not in df.columns:
        idx = df.columns.get_loc('cell_2') + 1
        df.insert(idx, 'sektör_2', df['cell_2'].astype(str).map(get_sector_label))
        df.insert(idx + 1, 'ortam_2', df['cell_2'].astype(str).map(get_cell_environment))
    return df


def _is_co_sector_by_id(cell_id_1: str, cell_id_2: str) -> bool:
    """Check if two cells are co-sector based on cell ID naming convention.
    Co-sector = same base site name AND same sector number (last letter group).
    These cells share the same physical antenna → must share PCI.

    IMPORTANT: For mixed indoor+outdoor sites (detected by detect_sector_groups),
    the naming convention is unreliable because indoor cells restart sector
    lettering from A.  In that case this function returns False, and the
    caller must rely on the cell_to_sector dict (azimuth-based) instead.
    """
    s1 = _extract_site_name(cell_id_1)
    s2 = _extract_site_name(cell_id_2)
    if s1 is None or s2 is None:
        return False
    if s1 != s2:
        return False
    # For mixed/indoor sites, naming convention is NOT reliable
    # → return False so callers fall through to cell_to_sector dict
    if s1 in _MIXED_INDOOR_SITES:
        return False
    sec1 = _extract_sector_number(cell_id_1)
    sec2 = _extract_sector_number(cell_id_2)
    if sec1 is None or sec2 is None:
        return False
    return sec1 == sec2


# ============================================================
# Sector Grouping  (hybrid: naming convention + azimuth)
# ============================================================
def detect_sector_groups(df, azimuth_tolerance=10.0, location_tolerance_m=50.0):
    """Group cells that belong to the same physical sector.

    Uses **site_id** column to identify same-site cells, then applies
    one of three rules per site:

    RULE 1 — BALANCED SITE (naming convention):
       Naming-convention sectors all have equal band count AND within
       each sector the azimuth spread among bands is < azimuth_tolerance.
       → Naming convention is reliable: co-sector by same sector number.
       → Sector numbering from naming convention.

    RULE 2 — UNBALANCED SITE (azimuth fallback):
       At least one naming-convention sector has a different band count.
       → Naming convention is unreliable (indoor cells may reuse outdoor
         letters at different azimuths).
       → Co-sector = cells within azimuth_tolerance of each other.
       → 1800-only cells with azimuth 0° or 360° are typically indoor.

    RULE 3 — PURE INDOOR SITE:
       ALL naming-convention sectors are 1800-only AND all cells on the
       site share the same azimuth (spread < azimuth_tolerance).
       → All cells are indoor.  Use naming convention for sector grouping
         (azimuths are unreliable, all the same).

    FALLBACK (cells not matching Turkish naming convention):
       Same physical location (≤ location_tolerance_m) AND same azimuth.

    Side effects:
       - Populates _MIXED_INDOOR_SITES for sites using RULE 2.
       - Populates _DETECTED_INDOOR_CELLS with indoor cell IDs.

    Returns:
        sector_groups: dict  sector_key → list of cell_ids
        cell_to_sector: dict  cell_id → sector_key
    """
    # Clear (don't reassign) so imported references stay valid
    _MIXED_INDOOR_SITES.clear()
    _DETECTED_INDOOR_CELLS.clear()

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return {}, {}

    sector_groups = {}   # sector_key → [cell_ids]
    cell_to_sector = {}  # cell_id → sector_key
    sector_counter = 0

    # Build list of cells with their coordinates and azimuth
    _has_sector_col = 'sector' in df.columns
    cells = []
    for _, r in df.iterrows():
        _sec_val = ''
        if _has_sector_col:
            sv = r.get('sector')
            if sv is not None and not (isinstance(sv, float) and pd.isna(sv)):
                _sec_val = str(sv).strip()
                if _sec_val.lower() in ('', 'nan', 'none'):
                    _sec_val = ''
        cells.append({
            'cell_id': str(r['cell_id']),
            'lat': r['latitude'],
            'lon': r['longitude'],
            'azimuth': r.get('azimuth', 0) or 0,
            'site_id': str(r.get('site_id', '')).strip(),
            'sector_col': _sec_val,
        })

    n = len(cells)
    loc_tol_km = location_tolerance_m / 1000.0
    handled = set()  # cell indices already assigned to a sector group

    def _az_diff(a, b):
        """Smallest angular difference between two azimuths."""
        d = abs(a - b) % 360
        return d if d <= 180 else 360 - d

    def _az_spread(indices):
        """Max azimuth spread among a set of cell indices."""
        if len(indices) <= 1:
            return 0
        azs = [cells[i]['azimuth'] for i in indices]
        max_diff = 0
        for i in range(len(azs)):
            for j in range(i + 1, len(azs)):
                d = _az_diff(azs[i], azs[j])
                if d > max_diff:
                    max_diff = d
        return max_diff

    # ── STEP 1: Group cells by site ────────────────────────────
    # Primary: site_id column.  Secondary: naming convention site name.
    # Build site_key → [cell_indices]
    site_groups = defaultdict(list)  # site_key → [cell_indices]
    naming_site_map = {}  # index → naming-convention site name (for logging)

    for i, c in enumerate(cells):
        # Use site_id if available
        sid = c['site_id']
        if sid and sid not in ('', 'nan', 'None', 'NaN'):
            site_groups[f'SID_{sid}'].append(i)
        else:
            # Fallback to naming convention site name
            sn = _extract_site_name(c['cell_id'])
            if sn:
                site_groups[f'NAME_{sn}'].append(i)
            # else: will be handled by location fallback later

        sn = _extract_site_name(c['cell_id'])
        if sn:
            naming_site_map[i] = sn

    # ── STEP 1b: Merge site_groups sharing the same physical site ─
    # In Huawei (and multi-vendor) data, each band has its own eNodeB with
    # a different site_id (e.g. EBU1701, TBU1701, LBU1701, ZBU1701).
    # These are all at the same physical site BU1701.  Merge them so that
    # co-sector cells across bands share the same PCI.
    _base_to_keys = defaultdict(list)  # naming base → [site_group_keys]
    for sg_key, sg_indices in list(site_groups.items()):
        for idx in sg_indices:
            sn = _extract_site_name(cells[idx]['cell_id'])
            if sn:
                _base_to_keys[sn].append(sg_key)
                break
    for _base, _keys in _base_to_keys.items():
        if len(_keys) <= 1:
            continue
        # Deduplicate (same key could appear via multiple cells)
        _unique_keys = list(dict.fromkeys(_keys))
        if len(_unique_keys) <= 1:
            continue
        _target = _unique_keys[0]
        for _other in _unique_keys[1:]:
            if _other in site_groups:
                site_groups[_target].extend(site_groups.pop(_other))

    # ── STEP 2: Per-site rule decision ─────────────────────────
    for site_key, site_indices in site_groups.items():
        if len(site_indices) < 1:
            continue

        # ── SECTOR COLUMN OVERRIDE ──────────────────────────────
        # If ALL cells at this site have a populated 'sector' column,
        # use that directly instead of the 3-rule naming/azimuth system.
        _all_have_sec = all(cells[i]['sector_col'] != '' for i in site_indices)
        if _all_have_sec:
            _sec_col_groups: Dict[str, list] = defaultdict(list)
            for i in site_indices:
                _sec_col_groups[cells[i]['sector_col']].append(i)
            for sec_val, indices in _sec_col_groups.items():
                sector_key = f"COL_{site_key}_SEC{sec_val}_S{sector_counter}"
                sector_counter += 1
                member_ids = [cells[idx]['cell_id'] for idx in indices]
                if len(indices) >= 2:
                    sector_groups[sector_key] = member_ids
                for cid in member_ids:
                    cell_to_sector[cid] = sector_key
            handled.update(site_indices)
            continue

        # Build naming-convention sectors for this site
        # sec_num → [cell_indices]
        nc_sectors = defaultdict(list)
        nc_matched = 0
        for i in site_indices:
            sec_num = _extract_sector_number(cells[i]['cell_id'])
            if sec_num is not None:
                nc_sectors[sec_num].append(i)
                nc_matched += 1

        # If naming convention doesn't match most cells, skip to fallback
        if nc_matched < len(site_indices) * 0.5:
            # Will be handled by location fallback
            continue

        # Analyse band distribution per naming-convention sector
        bands_per_sector = {}  # sec_num → set of band prefixes
        for sec_num, indices in nc_sectors.items():
            bands = set()
            for idx in indices:
                bp = _extract_band_prefix(cells[idx]['cell_id'])
                if bp:
                    bands.add(bp)
            bands_per_sector[sec_num] = bands

        band_counts = [len(b) for b in bands_per_sector.values()]
        all_1800_only = all(b == {'T'} for b in bands_per_sector.values())

        # ── RULE 3: Pure indoor site ────────────────────────
        # All sectors 1800-only AND all cells share same azimuth
        if all_1800_only:
            overall_spread = _az_spread(site_indices)
            if overall_spread < azimuth_tolerance:
                # RULE 3: Pure indoor — all cells are indoor
                # Use naming convention for sector grouping
                for i in site_indices:
                    _DETECTED_INDOOR_CELLS.add(cells[i]['cell_id'])

                for sec_num, indices in nc_sectors.items():
                    sector_key = f"NC_{site_key}_SEC{sec_num}_S{sector_counter}"
                    sector_counter += 1
                    member_ids = [cells[idx]['cell_id'] for idx in indices]
                    if len(indices) >= 2:
                        sector_groups[sector_key] = member_ids
                    for cid in member_ids:
                        cell_to_sector[cid] = sector_key
                handled.update(site_indices)
                continue
            else:
                # All 1800-only but different azimuths -
                # treat like naming convention (each sector has its own direction)
                # These are still indoor cells
                for i in site_indices:
                    _DETECTED_INDOOR_CELLS.add(cells[i]['cell_id'])

                for sec_num, indices in nc_sectors.items():
                    sector_key = f"NC_{site_key}_SEC{sec_num}_S{sector_counter}"
                    sector_counter += 1
                    member_ids = [cells[idx]['cell_id'] for idx in indices]
                    if len(indices) >= 2:
                        sector_groups[sector_key] = member_ids
                    for cid in member_ids:
                        cell_to_sector[cid] = sector_key
                handled.update(site_indices)
                continue

        # ── Check if balanced (equal band count across all sectors) ──
        balanced = (len(set(band_counts)) <= 1)

        if balanced:
            # ── RULE 1 check: also verify azimuth consistency ──
            # Within each naming-convention sector, all cells' azimuths
            # must be within azimuth_tolerance of each other.
            az_consistent = True
            for sec_num, indices in nc_sectors.items():
                if _az_spread(indices) >= azimuth_tolerance:
                    az_consistent = False
                    break

            if az_consistent:
                # ── RULE 1: Naming convention is reliable ──────
                for sec_num, indices in nc_sectors.items():
                    sector_key = f"NC_{site_key}_SEC{sec_num}_S{sector_counter}"
                    sector_counter += 1
                    member_ids = [cells[idx]['cell_id'] for idx in indices]
                    if len(indices) >= 2:
                        sector_groups[sector_key] = member_ids
                    for cid in member_ids:
                        cell_to_sector[cid] = sector_key
                handled.update(site_indices)
                continue
            # else: fall through to RULE 2 (azimuth inconsistent)

        # ── RULE 2: Unbalanced / azimuth-inconsistent → azimuth fallback ──
        # Extract naming site name for _MIXED_INDOOR_SITES
        for i in site_indices:
            sn = naming_site_map.get(i)
            if sn:
                _MIXED_INDOOR_SITES.add(sn)

        # Mark 1800-only cells with azimuth 0° or 360° as indoor
        for i in site_indices:
            bp = _extract_band_prefix(cells[i]['cell_id'])
            az = cells[i]['azimuth']
            if bp == 'T':
                # Check if this cell is 1800-only (no other band at same sector)
                sec_num = _extract_sector_number(cells[i]['cell_id'])
                is_only_1800 = True
                if sec_num is not None:
                    sector_bands = bands_per_sector.get(sec_num, set())
                    is_only_1800 = (sector_bands == {'T'})
                if is_only_1800 and (az <= azimuth_tolerance or _az_diff(az, 360) <= azimuth_tolerance):
                    _DETECTED_INDOOR_CELLS.add(cells[i]['cell_id'])

        # Indoor cells → group by naming convention sector number
        # (indoor cells share azimuth 0° with outdoor sector 1 but are
        #  logically separate sectors and cannot share PCI on same band)
        _indoor_nc = defaultdict(list)   # sec_num → [index, …]
        _indoor_pos = set()              # positions in site_indices
        for i_pos, ci in enumerate(site_indices):
            if cells[ci]['cell_id'] in _DETECTED_INDOOR_CELLS:
                sec_num = _extract_sector_number(cells[ci]['cell_id'])
                if sec_num is not None:
                    _indoor_nc[sec_num].append(ci)
                    _indoor_pos.add(i_pos)

        for sec_num, indoor_group in _indoor_nc.items():
            sector_key = f"NC_{site_key}_SEC{sec_num}_S{sector_counter}"
            sector_counter += 1
            member_ids = [cells[idx]['cell_id'] for idx in indoor_group]
            if len(indoor_group) >= 2:
                sector_groups[sector_key] = member_ids
            for cid in member_ids:
                cell_to_sector[cid] = sector_key

        # Remaining (outdoor) cells → group by azimuth proximity
        az_assigned = [False] * len(site_indices)
        for i_pos in _indoor_pos:
            az_assigned[i_pos] = True          # already handled above
        for i_pos in range(len(site_indices)):
            if az_assigned[i_pos]:
                continue
            ci = site_indices[i_pos]
            az_i = cells[ci]['azimuth']
            group = [ci]
            az_assigned[i_pos] = True
            for j_pos in range(i_pos + 1, len(site_indices)):
                if az_assigned[j_pos]:
                    continue
                cj = site_indices[j_pos]
                az_j = cells[cj]['azimuth']
                if _az_diff(az_i, az_j) <= azimuth_tolerance:
                    group.append(cj)
                    az_assigned[j_pos] = True

            # Register this azimuth group
            sector_key = f"AZ_{site_key}_az{int(az_i)}_S{sector_counter}"
            sector_counter += 1
            member_ids = [cells[idx]['cell_id'] for idx in group]
            if len(group) >= 2:
                sector_groups[sector_key] = member_ids
            for cid in member_ids:
                cell_to_sector[cid] = sector_key
        handled.update(site_indices)

    # ── FALLBACK: Location + azimuth for unhandled cells ─────────
    remaining = [i for i in range(n) if i not in handled]
    if remaining:
        # Union-Find for location clustering
        parent = {i: i for i in remaining}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        # Location fallback: cells within tolerance → same site
        bucket_size = 0.001  # ~111m
        buckets = defaultdict(list)
        for i in remaining:
            c = cells[i]
            bkey = (round(c['lat'] / bucket_size), round(c['lon'] / bucket_size))
            buckets[bkey].append(i)

        for bkey, idxs in buckets.items():
            candidates = list(idxs)
            for dlat in (-1, 0, 1):
                for dlon in (-1, 0, 1):
                    if dlat == 0 and dlon == 0:
                        continue
                    adj_key = (bkey[0] + dlat, bkey[1] + dlon)
                    candidates.extend(buckets.get(adj_key, []))
            for i in idxs:
                for j in candidates:
                    if i >= j:
                        continue
                    d = haversine_distance(cells[i]['lat'], cells[i]['lon'],
                                           cells[j]['lat'], cells[j]['lon'])
                    if d <= loc_tol_km:
                        _union(i, j)

        # Within each site cluster, sub-group by azimuth
        site_clusters = defaultdict(list)
        for i in remaining:
            site_clusters[_find(i)].append(i)

        for root_idx, members_idx in site_clusters.items():
            if len(members_idx) < 2:
                continue
            az_assigned = [False] * len(members_idx)
            for i_pos in range(len(members_idx)):
                if az_assigned[i_pos]:
                    continue
                ci = members_idx[i_pos]
                az_i = cells[ci]['azimuth']
                group = [ci]
                az_assigned[i_pos] = True
                for j_pos in range(i_pos + 1, len(members_idx)):
                    if az_assigned[j_pos]:
                        continue
                    cj = members_idx[j_pos]
                    az_j = cells[cj]['azimuth']
                    diff = abs(az_i - az_j)
                    if diff > 180:
                        diff = 360 - diff
                    if diff <= azimuth_tolerance:
                        group.append(cj)
                        az_assigned[j_pos] = True

                if len(group) >= 2:
                    site_label = cells[group[0]]['site_id'] or f'LOC{sector_counter}'
                    sector_key = f"{site_label}_az{int(az_i)}_S{sector_counter}"
                    sector_counter += 1
                    member_ids = [cells[idx]['cell_id'] for idx in group]
                    sector_groups[sector_key] = member_ids
                    for cid in member_ids:
                        cell_to_sector[cid] = sector_key

    return sector_groups, cell_to_sector


# ============================================================
# PRACH Config Index → Preamble Format (3GPP TS 36.211 Table 5.7.1-2 FDD)
# ============================================================
# Duplex mode by band, in MHz (the form the operator's data carries).
# 2600 is deliberately FDD: it is Band 7 in this network.  Band 38 shares the
# frequency and is TDD, so an operator using it must override with a `duplex`
# column.
DUPLEX_BY_BAND_MHZ = {
    700: 'FDD',    # B28 / n28
    800: 'FDD',    # B20 / n20
    900: 'FDD',    # B8  / n8
    1800: 'FDD',   # B3  / n3
    2100: 'FDD',   # B1  / n1
    2300: 'TDD',   # B40 / n40
    2600: 'FDD',   # B7  (B38 at the same frequency is TDD - override if used)
    3500: 'TDD',   # n78
    3700: 'TDD',   # n77
}


def cell_duplex(row):
    """FDD or TDD for a cell.

    An explicit `duplex` column always wins; otherwise it is derived from the
    band.  Deriving beats asking for it: a hand-filled duplex column drifts,
    while the band is already needed for the carrier key.
    """
    v = row.get('duplex')
    if v is not None:
        try:
            if not pd.isna(v):
                t = str(v).strip().upper()
                if t.startswith('T'):
                    return 'TDD'
                if t.startswith('F'):
                    return 'FDD'
        except (TypeError, ValueError):
            pass
    for key in ('band', 'band_mhz'):
        b = row.get(key)
        if b is None:
            continue
        try:
            if pd.isna(b):
                continue
            return DUPLEX_BY_BAND_MHZ.get(int(float(b)), 'FDD')
        except (TypeError, ValueError):
            continue
    return 'FDD'


def get_lte_preamble_format(config_index, duplex='FDD'):
    """PRACH configuration index -> preamble format.

    FDD, TS 36.211 Table 5.7.1-2 (frame structure type 1):
        0-15 -> 0, 16-31 -> 1, 32-47 -> 2, 48-63 -> 3.
        Format 4 does NOT exist in FDD.  v2 mapped 58-63 to format 4, which
        silently switched those cells to L_RA=139.

    TDD, TS 36.211 Table 5.7.1-4 (frame structure type 2):
        indices 0-57 are valid and 58-63 are N/A.  Format 4 occupies the top
        of the valid range, 48-57 — that is the only boundary that changes
        L_RA (139 instead of 839) and therefore the only one that affects Ncs,
        cell range and root demand.  The boundaries between formats 0-3 below
        it are reproduced from the same table; if a future reader needs them
        to be exact for a purpose other than L_RA, check them against the spec.
    """
    ci = int(config_index) if not pd.isna(config_index) else 0
    tbl = (LTE_PRACH_FORMAT_TDD if str(duplex).strip().upper().startswith('T')
           else LTE_PRACH_FORMAT_FDD)
    return tbl.get(ci)          # None = N/A for this duplex mode

# ============================================================
# Geo utilities
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1-a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon)*cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360

def is_in_antenna_coverage(src_lat, src_lon, src_az, src_bw, tgt_lat, tgt_lon):
    bearing = calculate_bearing(src_lat, src_lon, tgt_lat, tgt_lon)
    diff = abs(bearing - src_az)
    if diff > 180: diff = 360 - diff
    return diff <= src_bw / 2.0

def decompose_pci(pci):
    """PCI = 3*SSS + PSS → (PSS, SSS)"""
    return int(pci) % 3, int(pci) // 3

# ============================================================
# PRACH / RSI Calculations
# ============================================================
def ncs_table(technology='LTE', restricted=False, short=False, delta_f_ra_khz=None):
    """The zeroCorrelationZoneConfig -> Ncs table that applies to a cell.

    Every Ncs lookup goes through here, so derivation in both directions (zcz
    -> Ncs for Nokia, cell range -> zcz for Huawei) reads the same table.

      LTE formats 0-3        TS 36.211 T5.7.2-2  (unrestricted / type A / type B)
      LTE format 4 (L=139)   TS 36.211 T5.7.2-3  — its own table, not the NR one
      NR  1.25 kHz (fmt 0-2) TS 38.211 T6.3.3.1-5
      NR  5 kHz    (fmt 3)   TS 38.211 T6.3.3.1-6  — differs from 1.25 kHz
      NR  L=139   (A/B/C)    TS 38.211 T6.3.3.1-7

    Restricted sets are not defined for L_RA=139, so `restricted` is ignored
    there.  v2 ignored `short` for LTE (a format 4 cell got Ncs=26 where the
    standard says 12); v3 until 2026-09 ignored both `restricted` and
    delta_f_RA for NR.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    set_b = bool(restricted) and str(restricted).upper().endswith('B')
    if technology == 'NR':
        if short:
            return NR_NCS_SHORT
        if delta_f_ra_khz is not None and abs(float(delta_f_ra_khz) - 5.0) < 1e-9:
            if restricted:
                return NR_NCS_5KHZ_RESTRICTED_B if set_b else NR_NCS_5KHZ_RESTRICTED_A
            return NR_NCS_5KHZ
        if restricted:
            return NR_NCS_LONG_RESTRICTED_B if set_b else NR_NCS_LONG_RESTRICTED_A
        return NR_NCS_LONG
    if short:
        return LTE_NCS_FORMAT4                  # N/A for zcz >= 7
    if restricted:
        return LTE_NCS_RESTRICTED_B if set_b else LTE_NCS_RESTRICTED
    return LTE_NCS_UNRESTRICTED


def get_ncs(zcz, technology='LTE', restricted=False, short=False, delta_f_ra_khz=None):
    """Ncs for a zeroCorrelationZoneConfig; 0 when the table says N/A."""
    return ncs_table(technology, restricted, short, delta_f_ra_khz).get(int(zcz), 0)


def _margin_applies(nzc, tseq_us):
    """The 5.2 us + 2-sample margin is verified for L_RA=839 at 1.25 kHz only."""
    return int(nzc) == NZC_LONG and abs(float(tseq_us) - 800.0) < 1e-6


def cell_range_from_ncs(ncs, nzc=NZC_LONG, tseq_us=800.0):
    """Largest cell radius (km) an Ncs supports.

    Where the margin is verified (see PRACH_DELAY_SPREAD_US) the window loses
    n_g samples and tau_ds before it is turned into distance — this reproduces
    the vendor Ncs -> cell radius table.  Elsewhere it is the plain round trip.
    """
    if ncs == 0:
        return 0.0
    window_us = ncs * tseq_us / nzc
    if _margin_applies(nzc, tseq_us):
        window_us = ((ncs - PRACH_GUARD_SAMPLES) * tseq_us / nzc
                     - PRACH_DELAY_SPREAD_US)
    return max(window_us, 0.0) * 1e-6 * SPEED_OF_LIGHT / 2 / 1000

def cell_range_from_format(fmt, tech='LTE'):
    """Largest cell radius an LTE preamble FORMAT allows (km).

    The round trip has to fit in the cyclic prefix AND in the guard time left
    after the sequence (TS 36.211 T5.7.1-1: GT = subframes x 1 ms - T_CP -
    T_SEQ).  Formats 0, 1 and 2 are guard-time limited — v3 used T_CP alone
    and showed format 1 as 102.7 km where the guard time allows 77.3 km.
    Format 4 sits in UpPTS and is CP limited.
    """
    f = LTE_PREAMBLE_FORMATS.get(fmt, LTE_PREAMBLE_FORMATS[0])
    limit_us = f['tcp_us']
    if fmt != 4:
        guard_us = f['subframes'] * 1000.0 - f['tcp_us'] - f['tseq_us']
        limit_us = min(limit_us, guard_us)
    return limit_us * 1e-6 * SPEED_OF_LIGHT / 2 / 1000

def preambles_per_root(ncs, nzc=NZC_LONG):
    """Cyclic shifts per root in the UNRESTRICTED set: floor(N_ZC / N_CS).

    Only valid when restrictedSetConfig is 'unrestricted'.  For the restricted
    (high-speed) sets the count varies per root — see
    preambles_per_root_restricted().
    """
    return max(1, floor(nzc/ncs)) if ncs > 0 else 1

def roots_needed(num_preambles, ncs, nzc=NZC_LONG):
    """Roots needed in the UNRESTRICTED set."""
    return int(ceil(num_preambles / preambles_per_root(ncs, nzc)))


# ============================================================
# Restricted (high-speed) set — K-2
# ============================================================
# 3GPP TS 36.211 §5.7.2 and TS 38.211 §6.3.3.1.  In the restricted set the
# number of cyclic shifts a root yields is NOT floor(N_ZC/N_CS): it depends on
# d_u, the cyclic shift of the root, and therefore varies from root to root.
# Measured over all u for N_ZC=839: at N_CS=76 the unrestricted formula says 11
# shifts per root while the restricted set yields a median of 2 — the
# difference between reserving 6 roots and reserving 32.

def root_cyclic_shift(u, nzc=NZC_LONG):
    """d_u — the cyclic shift of physical root u.

    d_u = p if p < N_ZC/2 else N_ZC - p, where p is the modular inverse of u.
    """
    u = int(u) % nzc
    if u == 0:
        return 0
    p = pow(u, -1, nzc)
    return p if p < nzc / 2 else nzc - p


def preambles_per_root_restricted(u, ncs, nzc=NZC_LONG, set_type='A'):
    """Cyclic shifts yielded by physical root u in the restricted set.

    set_type 'A' implements TS 36.211 §5.7.2 / TS 38.211 restrictedSetTypeA
    exactly.  Type B adds further constraints that are not implemented; it
    falls back to the Type A value, which is an UPPER bound, so callers must
    treat a Type B result as approximate (see _prach_params: restricted_approx).
    """
    if ncs <= 0:
        return 1
    d_u = root_cyclic_shift(u, nzc)
    if ncs <= d_u < nzc / 3:
        n_shift = d_u // ncs
        d_start = 2 * d_u + n_shift * ncs
        n_group = nzc // d_start if d_start else 0
        n_shift_bar = max((nzc - 2 * d_u - n_group * d_start) // ncs, 0)
    elif nzc / 3 <= d_u <= (nzc - ncs) // 2:
        n_shift = (nzc - 2 * d_u) // ncs
        d_start = nzc - 2 * d_u + n_shift * ncs
        n_group = d_u // d_start if d_start else 0
        n_shift_bar = min(max((d_u - n_group * d_start) // ncs, 0), n_shift)
    else:
        return 0
    return n_shift * n_group + n_shift_bar


# Logical -> physical root order (TS 36.211 Table 5.7.2-4 for L=839,
# TS 38.211 Table 6.3.3.1-3).  This is an engineered permutation, not a
# formula: it CANNOT be derived (the first entries are 129, 710, 140, 699,
# 120, 719 ... whose d_u values are 13, 13, 6, 6, 7, 7 — not monotonic).
# Until it is supplied, restricted-set root counts cannot be resolved exactly,
# because how many logical indices a cell consumes depends on which physical
# roots those indices land on.  Load it with set_root_order().
ROOT_ORDER = {}          # nzc -> tuple(physical roots, indexed by logical index)


def set_root_order(order, nzc=NZC_LONG):
    """Install the logical->physical root order for a sequence length.

    `order` must list the physical root for every logical index, in order
    (838 entries for N_ZC=839, 138 for N_ZC=139).
    """
    order = tuple(int(x) for x in order)
    expected = nzc - 1
    if len(order) != expected:
        raise ValueError(f"N_ZC={nzc} icin {expected} girdi bekleniyor, "
                         f"{len(order)} verildi")
    if not all(1 <= x < nzc for x in order):
        raise ValueError("fiziksel kok degerleri 1..N_ZC-1 araliginda olmali")
    if len(set(order)) != len(order):
        raise ValueError("tabloda tekrar eden fiziksel kok var")
    ROOT_ORDER[nzc] = order
    return len(order)


def has_root_order(nzc=NZC_LONG):
    return nzc in ROOT_ORDER


# Install the spec tables at import: TS 36.211 Table 5.7.2-4 (L_RA=839,
# preamble formats 0-3) and Table 5.7.2-5 (L_RA=139, format 4).  With these
# present the restricted-set root demand is resolved exactly instead of being
# estimated from the median per-root yield.
set_root_order(ROOT_ORDER_839, NZC_LONG)
set_root_order(ROOT_ORDER_139, NZC_SHORT)


def roots_needed_for_cell(rsi_start, ncs, nzc=NZC_LONG, restricted=False,
                          n_preambles=64, set_type='A'):
    """Consecutive LOGICAL root indices this cell consumes.

    Unrestricted: constant per root, so ceil(64 / floor(N_ZC/N_CS)) — the start
    index is irrelevant.  Restricted: each root yields a different number of
    shifts, so the walk starts at rsi_start and accumulates until 64 preambles
    are reached, which makes the answer depend on WHERE the cell starts.

    Returns None when the restricted case cannot be resolved because the
    logical->physical root order has not been installed.
    """
    if ncs <= 0:
        return int(n_preambles)     # no cyclic shift: one preamble per root
    if not restricted:
        return roots_needed(n_preambles, ncs, nzc)
    order = ROOT_ORDER.get(nzc)
    if not order:
        return None
    m = len(order)
    start = int(rsi_start) % m
    total = 0
    for k in range(m):
        u = order[(start + k) % m]
        total += preambles_per_root_restricted(u, ncs, nzc, set_type)
        if total >= n_preambles:
            return k + 1
    return None                     # not even the whole space is enough


def restricted_roots_bounds(ncs, nzc=NZC_LONG, set_type='A'):
    """(best, typical, worst) root counts over every possible start index.

    Used to bound the answer when the root order is unknown: the walk cannot be
    performed, but the DISTRIBUTION of per-root yields is still computable, so
    a defensible reservation can be reported instead of the unrestricted
    formula's optimistic number.
    """
    if ncs <= 0:
        return (n := int(64)), n, n
    yields = sorted(preambles_per_root_restricted(u, ncs, nzc, set_type)
                    for u in range(1, nzc))
    if not yields or yields[-1] <= 0:
        return None, None, None
    best = int(ceil(64 / yields[-1]))
    med = yields[len(yields) // 2]
    typical = int(ceil(64 / med)) if med > 0 else None
    # worst case: take the smallest yields first until 64 preambles accumulate
    total = 0
    worst = None
    for i, y in enumerate(yields):
        total += y
        if total >= 64:
            worst = i + 1
            break
    return best, typical, worst

def rsi_overlap(rsi1, ncs1, rsi2, ncs2, nzc=NZC_LONG, npre=64, max_rsi=LTE_RSI_COUNT,
                r1=None, r2=None):
    """Do two cells' root-sequence ranges overlap?

    r1/r2 let the caller pass restricted-set root counts, which cannot be
    derived from Ncs alone.  Omitted, they fall back to the unrestricted
    formula.
    """
    if r1 is None:
        r1 = roots_needed(npre, ncs1, nzc)
    if r2 is None:
        r2 = roots_needed(npre, ncs2, nzc)
    s1 = {(rsi1+i)%max_rsi for i in range(r1)}
    s2 = {(rsi2+i)%max_rsi for i in range(r2)}
    return len(s1 & s2) > 0


def root_space_size(nzc=NZC_LONG):
    """Number of logical root indices = the wrap modulus: 838 (L=839), 138 (L=139)."""
    return (NZC_SHORT - 1) if int(nzc) == NZC_SHORT else (NZC_LONG - 1)


def cell_root_count(start, ncs, nzc=NZC_LONG, restricted=False, candidate=False):
    """Roots a cell consumes when its RSI is `start`.

    Unrestricted: a constant.  Restricted: walked from `start` (K-2).  When the
    walk cannot reach 64 preambles, a CANDIDATE start is unusable (None); for
    an existing assignment the typical bound is used — the same number
    _prach_params() reports, so the checkers and the detector agree.
    """
    if not restricted:
        return roots_needed(64, ncs, nzc)
    set_type = 'B' if str(restricted).upper().endswith('B') else 'A'
    rn = roots_needed_for_cell(start, ncs, nzc, restricted=True, set_type=set_type)
    if rn is None and not candidate:
        rn = restricted_roots_bounds(ncs, nzc, set_type)[1]
    return rn


def cell_root_set(start, ncs, nzc=NZC_LONG, restricted=False, candidate=False):
    """Logical root indices a cell occupies from `start`, wrapped in its own root
    space.  None when `start` is not a usable RSI for this cell."""
    mx = root_space_size(nzc)
    if start is None or int(start) < 0 or int(start) >= mx:
        return None
    rn = cell_root_count(int(start), ncs, nzc, restricted, candidate)
    if rn is None or rn > mx:
        return None
    return {(int(start) + i) % mx for i in range(rn)}


NCS_ADEQUACY_LOW = 0.5    # supported < 0.5 x D_p  -> YETERSİZ
NCS_ADEQUACY_HIGH = 2.0   # supported > 2.0 x D_p  -> AŞIRI (if a smaller zcz covers D_p)


def ncs_adequacy(df, nbr_attempts, technology='LTE', carrier_map=None,
                 percentile=90, low=NCS_ADEQUACY_LOW, high=NCS_ADEQUACY_HIGH):
    """Does each cell's Ncs cover the distance it actually serves?  (Ö-1)

    The serving distance is estimated from real handovers: D_p = the
    attempts-weighted p-th percentile of the distances to the cell's
    handover partners (same-site pairs, < 50 m, ignored).  A UE handing over
    from A to B is somewhere between A and B, so A's edge towards B lies
    between D/2 (equal cells) and D (A overshoots up to B).  The two flags
    use the ends of that band, so neither needs a guessed factor:

      YETERSİZ  supported range < low  x D_p — even the optimistic edge
                estimate is out of the Ncs window: preambles from the far
                edge alias into a neighbouring cyclic shift (RACH failures).
      AŞIRI     supported range > high x D_p, and a smaller zcz still covers
                the WHOLE D_p — Ncs is oversized and wastes root sequences.
                The suggested zcz is the smallest that covers D_p, i.e. the
                conservative end of the band.
      UYGUN     otherwise.
      HO verisi yok  no handover partner with attempts > 0.

    Returns a DataFrame, one row per cell.
    """
    technology = norm_tech(technology)
    carrier_map = carrier_map or build_carrier_map(df)
    ids = df['cell_id'].astype(str).tolist()
    lat = dict(zip(ids, pd.to_numeric(df['latitude'], errors='coerce')))
    lon = dict(zip(ids, pd.to_numeric(df['longitude'], errors='coerce')))

    partners = defaultdict(list)          # cell -> [(km, attempts, other)]
    for pair, w in (nbr_attempts or {}).items():
        try:
            a, b = (str(x) for x in pair)
            w = float(w)
        except (TypeError, ValueError):
            continue
        if not w or w <= 0 or a not in lat or b not in lat:
            continue
        if pd.isna(lat[a]) or pd.isna(lat[b]):
            continue
        d = haversine_distance(lat[a], lon[a], lat[b], lon[b])
        if d < 0.05:
            continue
        partners[a].append((d, w, b))
        partners[b].append((d, w, a))

    def _wpct(items, p):
        items = sorted(items)
        cum = np.cumsum([w for _, w, _ in items])
        k = int(np.searchsorted(cum, p / 100.0 * cum[-1]))
        return items[min(k, len(items) - 1)][0]

    rows = []
    for cid, (_, row) in zip(ids, df.iterrows()):
        rsi_v = row.get('rsi')
        rsi_v = int(rsi_v) if rsi_v is not None and not pd.isna(rsi_v) else None
        p = _prach_params(row, technology, rsi=rsi_v)
        ncs, nzc, tseq = p['ncs'], p['nzc'], p['tseq_us']
        supported = cell_range_from_ncs(ncs, nzc, tseq)
        roots_now = p['roots_needed']
        rec = {'cell_id': cid, 'carrier': carrier_map.get(cid, CARRIER_UNKNOWN),
               'zcz': p['zcz'], 'ncs': ncs, 'ncs_range_km': round(supported, 2),
               'roots_now': roots_now}
        cr = row.get('cell_range')
        if cr is not None and not pd.isna(cr):
            rec['cell_range_input_m'] = int(cr)
        items = partners.get(cid, [])
        if not items:
            rec.update(status='HO verisi yok', ho_partners=0)
            rows.append(rec)
            continue
        dp = _wpct(items, percentile)
        far = max(items)
        rec.update(ho_partners=len(items),
                   ho_attempts=int(sum(w for _, w, _ in items)),
                   d50_km=round(_wpct(items, 50), 2),
                   **{f'd{percentile}_km': round(dp, 2)},
                   farthest_ho_km=round(far[0], 2), farthest_ho_cell=far[2],
                   farthest_ho_attempts=int(far[1]))
        status = 'UYGUN'
        if supported < low * dp:
            status = 'YETERSİZ'
        elif supported > high * dp:
            table = ncs_table(technology, p['restricted'], p['is_short'], p['delta_f_ra_khz'])
            for z, v in sorted(table.items(), key=lambda kv: kv[1]):
                if v and cell_range_from_ncs(v, nzc, tseq) >= dp - 1e-9:
                    if v < ncs:
                        rn = cell_root_count(rsi_v or 0, v, nzc, p['restricted'])
                        status = 'AŞIRI'
                        rec.update(suggested_zcz=z, suggested_ncs=v,
                                   suggested_range_km=round(cell_range_from_ncs(v, nzc, tseq), 2),
                                   roots_suggested=rn)
                    break
        rec['status'] = status
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        order = {'YETERSİZ': 0, 'AŞIRI': 1, 'UYGUN': 2, 'HO verisi yok': 3}
        out = out.sort_values('status', key=lambda s: s.map(order), kind='stable')
        out = out.reset_index(drop=True)
        for c in ('suggested_zcz', 'suggested_ncs', 'roots_suggested', 'ho_partners',
                  'ho_attempts', 'farthest_ho_attempts', 'cell_range_input_m'):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce').astype('Int64')
        lead = ['cell_id', 'status', 'carrier', 'zcz', 'ncs', 'ncs_range_km']
        out = out[[c for c in lead if c in out.columns]
                  + [c for c in out.columns if c not in lead]]
    return out


def _haversine_vec(lat0, lon0, lats, lons):
    """Great-circle distance (km) from one point to many, vectorised."""
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    la, lo = np.radians(np.asarray(lats, float)), np.radians(np.asarray(lons, float))
    a = (np.sin((la - la0) / 2) ** 2
         + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def rsi_reuse_distances(df, rsi_map=None, technology='LTE', carrier_map=None,
                        cell_to_sector=None):
    """RSI reuse distance of every cell.

    For each cell: the distance (km) to the nearest OTHER cell on the same
    carrier and the same L_RA that occupies at least one of its root
    sequences, and that cell's id.  Co-sector cells are skipped (they share
    the RSI by design and sit on other carriers).  A cell whose roots nobody
    else uses gets inf.  A neighbour pair that collides shows up here as a
    small distance, so this is the one number that makes an RSI plan
    reviewable: how close does the nearest reuse come.

    rsi_map: cell -> RSI to evaluate (default: the 'rsi' column).
    Returns {cell_id: (km, partner_id or None)}.
    """
    technology = norm_tech(technology)
    carrier_map = carrier_map or build_carrier_map(df)
    cell_to_sector = cell_to_sector or {}
    ids = df['cell_id'].astype(str).tolist()
    if rsi_map is None:
        rsi_map = dict(zip(ids, df['rsi'])) if 'rsi' in df.columns else {}
    lat = dict(zip(ids, pd.to_numeric(df['latitude'], errors='coerce')))
    lon = dict(zip(ids, pd.to_numeric(df['longitude'], errors='coerce')))

    roots, space = {}, {}
    for cid, (_, row) in zip(ids, df.iterrows()):
        v = rsi_map.get(cid)
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        p = _prach_params(row, technology, rsi=v)
        s = cell_root_set(v, p['ncs'], p['nzc'], p['restricted'])
        if s:
            roots[cid] = s
            space[cid] = (carrier_map.get(cid, CARRIER_UNKNOWN), p['nzc'])

    by_root = defaultdict(list)          # (carrier, nzc, root) -> cells
    for cid, s in roots.items():
        for r in s:
            by_root[space[cid] + (r,)].append(cid)

    out = {}
    for cid, s in roots.items():
        sec = cell_to_sector.get(cid)
        partners = {x for r in s for x in by_root[space[cid] + (r,)]
                    if x != cid and (sec is None or cell_to_sector.get(x) != sec)}
        partners = [x for x in partners
                    if pd.notna(lat.get(x)) and pd.notna(lon.get(x))]
        if not partners or pd.isna(lat.get(cid)) or pd.isna(lon.get(cid)):
            out[cid] = (float('inf'), None)
            continue
        d = _haversine_vec(lat[cid], lon[cid], [lat[x] for x in partners],
                           [lon[x] for x in partners])
        i = int(np.argmin(d))
        out[cid] = (float(d[i]), partners[i])
    return out

# ============================================================
# Huawei cellRange → zcz Reverse Mapping
# ============================================================
def derive_zcz_from_cell_range(cell_range_m, technology='LTE',
                                preamble_format=0, restricted=False,
                                duplex='FDD', scs_khz=None, band_mhz=None,
                                return_exceeded=False):
    """Given a cell range in metres (e.g. Huawei cellRadius), find the
    smallest zeroCorrelationZoneConfig whose Ncs covers that range.

    "Covers" is decided by cell_range_from_ncs(), so this is its exact inverse:
    the same table, the same delay-spread margin.  Feeding a reported range
    back in returns the same zcz.  (It used to recompute Ncs from the range
    with a float division, and 26.000000000000004 > 26 pushed an exact
    boundary one zcz too high.)

    Returns (zcz, ncs), or (zcz, ncs, exceeded) with return_exceeded=True.
    `exceeded` means no configurable Ncs reaches the range: the largest one is
    returned, and the caller should say so instead of planning silently for a
    smaller cell than the operator configured.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    cell_range_km = float(cell_range_m) / 1000.0

    # Determine Nzc, the sequence window and the Ncs table
    if technology == 'NR':
        _pcfg = int(preamble_format) if isinstance(preamble_format, (int, float)) else 0
        is_short, nzc, _dfra, _lbl = get_nr_preamble_info(
            _pcfg, duplex, nr_is_fr2(band_mhz, scs_khz))
        if is_short:
            _dfra = nr_short_delta_f_khz(scs_khz, band_mhz)
    else:
        fmt = get_lte_preamble_format(
            int(preamble_format) if not pd.isna(preamble_format) else 0, duplex)
        if fmt is None:
            fmt = 0                     # N/A index: same fallback as _prach_params
        is_short = (fmt == 4)
        nzc = NZC_SHORT if is_short else NZC_LONG
        _dfra = LTE_DELTA_F_RA_KHZ.get(fmt, 1.25)
    tseq_us = sequence_window_us(_dfra)
    table = ncs_table(technology, restricted, is_short, _dfra)

    if cell_range_km <= 0:
        return (5, table.get(5, 0), False) if return_exceeded else (5, table.get(5, 0))

    # Smallest configurable Ncs whose supported range reaches the cell.  Ncs=0
    # (no cyclic shift) is skipped: it is not a small-cell setting.
    candidates = sorted(((v, z) for z, v in table.items() if v and v > 0))
    best_ncs, best_zcz = candidates[-1]
    exceeded = True
    for ncs_val, zcz_val in candidates:
        if cell_range_from_ncs(ncs_val, nzc, tseq_us) >= cell_range_km - 1e-9:
            best_ncs, best_zcz, exceeded = ncs_val, zcz_val, False
            break

    if return_exceeded:
        return best_zcz, best_ncs, exceeded
    return best_zcz, best_ncs


RESTRICTED_SET_VALUES = ('unrestricted', 'typeA', 'typeB')


def restricted_set_config(row):
    """'unrestricted' | 'typeA' | 'typeB' for a cell.

    LTE Rel-13 and NR both define TWO restricted sets, and they diverge from
    zcz=11 (type A: 128, 158, 202, 237; type B: 118, 137, then N/A).  A boolean
    high-speed flag cannot express that, so the column is read as a value:
    NR calls it restrictedSetConfig, LTE calls it highSpeedFlag, and an
    operator carrying type B needs to be able to say so.
    """
    hs = row.get('restricted_set_config')
    if hs is None:
        hs = row.get('high_speed')
    if hs is None:
        return 'unrestricted'
    try:
        if pd.isna(hs):
            return 'unrestricted'
    except (TypeError, ValueError):
        pass
    t = str(hs).strip().lower().replace('-', '').replace('_', '').replace(' ', '')
    if t in ('', '0', '0.0', 'false', 'no', 'hayir', 'off', 'none',
             'unrestricted', 'unrestrictedset'):
        return 'unrestricted'
    if t.endswith('b') or t in ('2', '2.0', 'restrictedsettypeb', 'typeb'):
        return 'typeB'
    return 'typeA'


def _row_restricted(row):
    """Backward-compatible view of restricted_set_config().

    Returns False for unrestricted, otherwise the set name — which is truthy
    and is also what get_ncs() wants, so `restricted=_row_restricted(row)`
    keeps working and now carries the type through.
    """
    cfg = restricted_set_config(row)
    return False if cfg == 'unrestricted' else cfg


def _effective_zcz(row, technology='LTE'):
    """Return the effective zeroCorrelationZoneConfig for a cell row.

    Priority: if 'cell_range' column has a valid positive value → derive zcz
    from cellRange (Huawei mode).  Otherwise use explicit zero_correlation_zone.
    High-speed (restricted set) cells use the restricted Ncs table.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    cr = row.get('cell_range')
    if cr is not None and not pd.isna(cr) and float(cr) > 0:
        pcfg = int(row.get('prach_config_index', 0) or 0)
        zcz, _ = derive_zcz_from_cell_range(
            float(cr), technology, pcfg, restricted=_row_restricted(row),
            duplex=cell_duplex(row), scs_khz=row.get('msg1_scs_khz'),
            band_mhz=row.get('band') if row.get('band') is not None
            else row.get('band_mhz'))
        return zcz
    zcz = row.get('zero_correlation_zone', 5)
    if pd.isna(zcz):
        zcz = 5
    return int(zcz)

def _prach_params(row, technology='LTE', rsi=None):
    """Resolve every PRACH quantity for one cell row — the single derivation point.

    Preamble format, L_RA (Nzc), the effective zeroCorrelationZoneConfig, Ncs,
    the cyclic-shift window and the resulting root-sequence demand are all
    computed here.  Conflict detection, RSI planning, the new-cell finder and
    the UI panel all read from this function, so they can never disagree about
    a cell the way v2's six copies of this logic did.

    `rsi` matters only for the restricted (high-speed) set, where each root
    yields a different number of cyclic shifts and the count therefore depends
    on where the cell starts in the logical root order.

    Returns dict:
        technology, prach_config_index, preamble_format, is_short, nzc, zcz,
        restricted, ncs, tseq_us, max_rsi, preambles_per_root, roots_needed,
        roots_exact, roots_min, roots_max, feasible
    """
    technology = norm_tech(technology)
    pcfg = int(row.get('prach_config_index', 0) or 0)
    zcz = _effective_zcz(row, technology)
    restricted = _row_restricted(row)

    duplex = None
    invalid_pcfg = False
    if technology == 'NR':
        _band = row.get('band') if row.get('band') is not None else row.get('band_mhz')
        duplex = cell_duplex(row)
        _fr2 = nr_is_fr2(_band, row.get('msg1_scs_khz'))
        is_short, nzc, dfra, fmt = get_nr_preamble_info(pcfg, duplex, _fr2)
        # Index not defined in the table that applies (FDD 256-262): format 0
        # fallback above, flagged here — same treatment as LTE N/A indices.
        invalid_pcfg = nr_preamble_format(pcfg, duplex, _fr2) is None
        if is_short:
            # L_RA=139 window is set by msg1-SubcarrierSpacing (15*2^mu kHz).
            dfra = nr_short_delta_f_khz(row.get('msg1_scs_khz'), _band)
    else:
        duplex = cell_duplex(row)
        fmt = get_lte_preamble_format(pcfg, duplex)
        if fmt is None:
            # prach-ConfigIndex is N/A for this duplex mode (TDD 58-63).
            # Fall back to format 0 so the run continues, and flag it.
            fmt = 0
            invalid_pcfg = True
        is_short = (fmt == 4)          # LTE format 4 is TDD-only, L_RA = 139
        nzc = NZC_SHORT if is_short else NZC_LONG
        dfra = LTE_DELTA_F_RA_KHZ.get(fmt, 1.25)
    # The cyclic-shift window is ONE sequence, not the repeated total.
    tseq_us = sequence_window_us(dfra)

    ncs = get_ncs(zcz, technology, restricted=restricted, short=is_short,
                  delta_f_ra_khz=dfra)

    # Huawei mode: does the largest configurable Ncs actually reach the
    # configured cellRadius?  If not, the plan is for a smaller cell than the
    # operator set up, and that must be visible rather than silently clamped.
    cell_range_exceeded = False
    _cr = row.get('cell_range')
    if _cr is not None and not pd.isna(_cr) and float(_cr) > 0 and ncs:
        cell_range_exceeded = (
            cell_range_from_ncs(ncs, nzc, tseq_us) * 1000.0 < float(_cr) - 1e-6)

    # Root demand.  Unrestricted is a constant per root; restricted is not, so
    # it needs the logical->physical root order to be resolved exactly.  When
    # that table is not installed we fall back to the median-per-root estimate
    # and flag it, rather than silently reporting the unrestricted number,
    # which understates the demand by a factor of 3-5.
    roots_exact = True
    roots_min = roots_max = None
    feasible = True
    if restricted:
        _set_type = 'B' if str(restricted).upper().endswith('B') else 'A'
        rn = roots_needed_for_cell(rsi if rsi is not None else 0, ncs, nzc,
                                   restricted=True, set_type=_set_type)
        if rn is None:
            roots_exact = False
            roots_min, _typ, roots_max = restricted_roots_bounds(ncs, nzc, _set_type)
            if _typ is None:
                feasible = False
                rn = None
            else:
                rn = _typ
        ppr = preambles_per_root_restricted(1, ncs, nzc, _set_type)
    else:
        rn = roots_needed(64, ncs, nzc)
        ppr = preambles_per_root(ncs, nzc)

    return {
        'technology': technology,
        'prach_config_index': pcfg,
        'preamble_format': fmt,
        'is_short': is_short,
        'nzc': nzc,
        'zcz': zcz,
        'restricted': restricted,
        'restricted_set': restricted_set_config(row),
        'duplex': duplex,
        'delta_f_ra_khz': dfra,
        'invalid_prach_config': invalid_pcfg,
        'ncs': ncs,
        'tseq_us': tseq_us,
        'max_rsi': (NZC_SHORT - 1) if is_short else (NZC_LONG - 1),
        'preambles_per_root': ppr,
        'roots_needed': rn,
        'roots_exact': roots_exact,
        'roots_min': roots_min,
        'roots_max': roots_max,
        'feasible': feasible,
        'cell_range_exceeded': cell_range_exceeded,
    }


def compute_cell_prach_info(row, technology='LTE'):
    """UI-facing view of a cell's PRACH configuration.

    Thin wrapper over _prach_params so the displayed Ncs / roots / cell range
    are always the same numbers the analysis and the planners use.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    p = _prach_params(row, technology)
    # cell_range_from_format() only knows the LTE T_CP table; for NR it would
    # return a meaningless number, so report nothing rather than something wrong.
    fmt_km = (round(cell_range_from_format(p['preamble_format'], technology), 2)
              if technology == 'LTE' else None)
    # Huawei mode: does the configured cellRadius even fit the preamble format?
    # When prach_config_index is missing it defaults to 0 (format 0, 14.5 km),
    # so a 38 km cell is a sign the index is missing, not that the cell is
    # wrong.  RSI is unaffected — formats 0-3 share one Ncs table and window.
    _cr = row.get('cell_range')
    exceeds_format = bool(
        fmt_km is not None and _cr is not None and not pd.isna(_cr)
        and float(_cr) > fmt_km * 1000.0)
    return {
        'preamble_format': p['preamble_format'], 'ncs': p['ncs'], 'nzc': p['nzc'],
        'effective_zcz': p['zcz'],
        'restricted': p['restricted'],
        'roots_exact': p['roots_exact'],
        'roots_min': p['roots_min'], 'roots_max': p['roots_max'],
        'feasible': p['feasible'],
        'cell_range_ncs_km': round(
            cell_range_from_ncs(p['ncs'], p['nzc'], p['tseq_us']), 2),
        'cell_range_format_km': fmt_km,
        'preambles_per_root': p['preambles_per_root'],
        'roots_needed': p['roots_needed'],
        'cell_range_exceeded': p['cell_range_exceeded'],
        'cell_range_exceeds_format': exceeds_format}

# ============================================================
# Neighbor Discovery
# ============================================================
def find_neighbors(df, radius_km, use_antenna=True, default_bw=65.0,
                    include_intra_site=True, external_neighbors=None):
    """Discover neighbor relations using multiple methods:

    1. **Distance + Antenna** (default): cells within radius_km AND in
       antenna coverage (azimuth ± beamwidth/2) are neighbors.
    2. **Intra-site** (include_intra_site=True): cells sharing the same
       site_id are ALWAYS neighbors regardless of distance/antenna.
       This catches same-site PCI collision which is critical.
    3. **External neighbor list** (external_neighbors): dict or DataFrame
       of explicit neighbor pairs from real ANR/handover data.
       These are ALWAYS added regardless of distance/antenna.
       If the external list includes an 'attempts' column, those
       handover attempt counts are stored per pair.

    Returns:
        nbrs: dict  cell_id → set of neighbor cell_ids
        nbr_sources: dict  (cell_a, cell_b) tuple → set of source labels
        nbr_attempts: dict  (cell_a, cell_b) tuple → int attempt count
                      (only pairs from external list with attempt data)
    """
    nbrs = defaultdict(set)
    nbr_sources = defaultdict(set)  # track WHY each pair is a neighbor
    nbr_attempts = {}              # (sorted pair) → attempt count
    n = len(df)
    lats, lons = df['latitude'].values, df['longitude'].values
    ids = df['cell_id'].values
    azs = df['azimuth'].values if 'azimuth' in df.columns else np.zeros(n)
    bws = df['beamwidth'].fillna(default_bw).values if 'beamwidth' in df.columns else np.full(n, default_bw)

    # --- Method 1: Distance + Antenna (spatial-bucket accelerated) ---
    # Buckets must be at least radius_km wide in BOTH directions, otherwise a
    # pair inside the radius can land two buckets apart and never be compared.
    # A degree of latitude is ~111 km everywhere, but a degree of longitude is
    # 111·cos(lat) — 87.7 km at Samsun's latitude.  Sizing the longitude bucket
    # as radius/111 made it only 79 % as wide as it needed to be, and 3.4 % of
    # the true within-radius pairs were silently dropped, all of them
    # east-west.  A missed pair is worse than a mis-scored one: the tool
    # neither reports its conflicts nor plans around it.
    _bucket_deg = max(radius_km / 111.0, 0.001)
    _mean_lat = float(np.nanmean(lats)) if n else 0.0
    _cos_lat = max(cos(radians(_mean_lat)), 0.15)   # guard near the poles
    _bucket_lon_deg = max(radius_km / (111.0 * _cos_lat), 0.001)
    _grid: dict = defaultdict(list)  # (brow, bcol) → [cell_index]
    for i in range(n):
        brow = int(lats[i] / _bucket_deg)
        bcol = int(lons[i] / _bucket_lon_deg)
        _grid[(brow, bcol)].append(i)

    _checked: set = set()  # avoid double-checking pairs
    for (brow, bcol), cell_indices in _grid.items():
        # Collect candidates from this bucket + 8 adjacent buckets
        candidates: list = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                adj = (brow + dr, bcol + dc)
                if adj in _grid:
                    candidates.extend(_grid[adj])
        for i in cell_indices:
            for j in candidates:
                if i >= j:
                    continue
                _pair_ij = (i, j)
                if _pair_ij in _checked:
                    continue
                _checked.add(_pair_ij)
                d = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if d <= radius_km:
                    if use_antenna:
                        ok = (is_in_antenna_coverage(lats[i],lons[i],azs[i],bws[i],lats[j],lons[j]) or
                              is_in_antenna_coverage(lats[j],lons[j],azs[j],bws[j],lats[i],lons[i]))
                        if ok:
                            nbrs[ids[i]].add(ids[j]); nbrs[ids[j]].add(ids[i])
                            pair = tuple(sorted([ids[i], ids[j]]))
                            nbr_sources[pair].add('mesafe+anten')
                    else:
                        nbrs[ids[i]].add(ids[j]); nbrs[ids[j]].add(ids[i])
                        pair = tuple(sorted([ids[i], ids[j]]))
                        nbr_sources[pair].add('mesafe')

    # --- Method 2: Intra-site (same site_id OR co-located) ---
    if include_intra_site:
        # 2a: same site_id
        if 'site_id' in df.columns:
            site_groups = df.groupby('site_id')['cell_id'].apply(list).to_dict()
            for site_id, cells in site_groups.items():
                for i in range(len(cells)):
                    for j in range(i+1, len(cells)):
                        nbrs[cells[i]].add(cells[j])
                        nbrs[cells[j]].add(cells[i])
                        pair = tuple(sorted([cells[i], cells[j]]))
                        nbr_sources[pair].add('aynı-site')
        # 2b: co-located cells (distance < 50m) — catches multi-band sites
        #     with different site_id prefixes (e.g. EAS0425 vs TAS0425)
        co_loc_km = 0.05  # 50 meters
        _coloc_bucket = 0.001  # ~111m per 0.001° latitude
        _coloc_grid: dict = defaultdict(list)
        for i in range(n):
            bkey = (round(lats[i] / _coloc_bucket), round(lons[i] / _coloc_bucket))
            _coloc_grid[bkey].append(i)
        for bkey, idxs in _coloc_grid.items():
            cands = list(idxs)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    adj = (bkey[0] + dr, bkey[1] + dc)
                    cands.extend(_coloc_grid.get(adj, []))
            for i in idxs:
                for j in cands:
                    if i >= j:
                        continue
                    d = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                    if d <= co_loc_km:
                        pair = tuple(sorted([ids[i], ids[j]]))
                        if pair not in nbr_sources:
                            nbrs[ids[i]].add(ids[j]); nbrs[ids[j]].add(ids[i])
                            nbr_sources[pair].add('aynı-konum')

    # --- Method 3: External neighbor list ---
    if external_neighbors is not None:
        if isinstance(external_neighbors, pd.DataFrame):
            # Expect columns: cell_1, cell_2  (or source, target / neighbor_1, neighbor_2)
            c1_col, c2_col = None, None
            cols_lower = {c.lower().strip(): c for c in external_neighbors.columns}
            for a, b in [('cell_1','cell_2'), ('source','target'),
                         ('neighbor_1','neighbor_2'), ('cell_a','cell_b'),
                         ('hucre_1','hucre_2'), ('serving_cell','neighbor_cell')]:
                if a in cols_lower and b in cols_lower:
                    c1_col, c2_col = cols_lower[a], cols_lower[b]
                    break
            # Detect attempts column
            from data_handler import resolve_attempt_column
            att_col, _att_how = resolve_attempt_column(external_neighbors.columns)
            if c1_col and c2_col:
                valid_ids = set(ids)
                for _, row in external_neighbors.iterrows():
                    ca, cb = str(row[c1_col]).strip(), str(row[c2_col]).strip()
                    if ca in valid_ids and cb in valid_ids and ca != cb:
                        nbrs[ca].add(cb); nbrs[cb].add(ca)
                        pair = tuple(sorted([ca, cb]))
                        nbr_sources[pair].add('harici-komşuluk')
                        # Store attempt count (sum if both directions exist)
                        if att_col is not None:
                            att_val = row.get(att_col)
                            if att_val is not None and not pd.isna(att_val):
                                att_int = int(float(att_val))
                                nbr_attempts[pair] = nbr_attempts.get(pair, 0) + att_int

    return nbrs, nbr_sources, nbr_attempts

# ============================================================
# Conflict Detection
# ============================================================
def _cell_pair_info(c, nb, loc_map, az_map, cell_to_sector):
    """Return (distance_km, is_facing, is_co_sector) for a cell pair."""
    # Check co-sector via naming convention (PRIMARY — deterministic)
    co = _is_co_sector_by_id(str(c), str(nb))
    # Fallback: pre-computed dict (from detect_sector_groups)
    if not co:
        co = (cell_to_sector.get(str(c), '__A') == cell_to_sector.get(str(nb), '__B')
              and cell_to_sector.get(str(c)) is not None)
    lc, ln = loc_map.get(c), loc_map.get(nb)
    dist = round(haversine_distance(lc[0],lc[1],ln[0],ln[1]),3) if lc and ln else None
    facing = False
    if lc and ln:
        az_c = az_map.get(c, 0) or 0
        az_nb = az_map.get(nb, 0) or 0
        facing = (is_in_antenna_coverage(lc[0],lc[1],az_c,120,ln[0],ln[1]) and
                  is_in_antenna_coverage(ln[0],ln[1],az_nb,120,lc[0],lc[1]))
    return dist, facing, co

def _build_loc_az_maps(df):
    loc = {r['cell_id']:(r['latitude'],r['longitude']) for _,r in df.iterrows()}
    az = dict(zip(df['cell_id'], df['azimuth'].fillna(0))) if 'azimuth' in df.columns else {}
    return loc, az


def build_co_site_set(df, cell_to_sector=None, tolerance_km=0.05):
    """Build a set of frozenset({cell_a, cell_b}) for co-site (non-co-sector) pairs.
    Co-site = within tolerance_km (default 50m) but NOT co-sector."""
    if cell_to_sector is None:
        cell_to_sector = {}
    locs = {str(r['cell_id']): (r['latitude'], r['longitude'])
            for _, r in df.iterrows()}
    result: set = set()
    ids = list(locs.keys())
    # Bucket by rounded lat/lon for O(n) average instead of O(n²)
    from collections import defaultdict as _dd
    bucket_size = 0.001  # ~111m per 0.001° latitude
    buckets = _dd(list)
    for cid in ids:
        lat, lon = locs[cid]
        bkey = (round(lat / bucket_size), round(lon / bucket_size))
        buckets[bkey].append(cid)
    # Check within same bucket and adjacent buckets
    for bkey, cells in buckets.items():
        # Gather candidates: this bucket + 8 adjacent
        candidates = list(cells)
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                if dlat == 0 and dlon == 0:
                    continue
                adj = (bkey[0] + dlat, bkey[1] + dlon)
                candidates.extend(buckets.get(adj, []))
        seen = set()
        for a in cells:
            for b in candidates:
                if a >= b:
                    continue
                pair = frozenset((a, b))
                if pair in seen:
                    continue
                seen.add(pair)
                d = haversine_distance(locs[a][0], locs[a][1],
                                       locs[b][0], locs[b][1])
                if d <= tolerance_km:
                    # Exclude co-sector pairs (naming convention is primary)
                    if _is_co_sector_by_id(a, b):
                        continue
                    sa = cell_to_sector.get(a)
                    sb = cell_to_sector.get(b)
                    if sa is not None and sa == sb:
                        continue  # co-sector, not co-site
                    result.add(pair)
    return result


def detect_collisions(df, neighbors, pci_col='pci', cell_to_sector=None, nbr_attempts=None,
                      _loc_az_cache=None, carrier_map=None):
    if cell_to_sector is None: cell_to_sector = {}
    if nbr_attempts is None: nbr_attempts = {}
    if carrier_map is None: carrier_map = {}
    pm = dict(zip(df['cell_id'], df[pci_col]))
    loc_map, az_map = _loc_az_cache if _loc_az_cache else _build_loc_az_maps(df)
    rows, seen = [], set()
    for c, ns in neighbors.items():
        cp = pm.get(c)
        if cp is None or pd.isna(cp): continue
        for nb in ns:
            np_ = pm.get(nb)
            if np_ is None or pd.isna(np_): continue
            if not same_carrier(carrier_map, c, nb):
                continue  # different carrier -> no physical interaction
            p = tuple(sorted([c, nb]))
            if int(cp)==int(np_) and p not in seen:
                seen.add(p)
                dist, facing, co = _cell_pair_info(c, nb, loc_map, az_map, cell_to_sector)
                if co:
                    continue  # co-sector cells share PCI by design – skip
                att = nbr_attempts.get(p, '')
                rows.append({'cell_1':c,'cell_2':nb,'pci':int(cp),
                    'distance_km': dist, 'facing': '✅' if facing else '❌',
                    'ho_attempts': att,
                    'carrier': carrier_map.get(str(c), ''),
                    'type':'COLLISION','severity':'CRITICAL',
                    'description':f'Same PCI {int(cp)} on neighboring cells'})
    return enrich_df_with_sector_info(pd.DataFrame(rows))

def detect_confusions(df, neighbors, pci_col='pci', cell_to_sector=None, nbr_attempts=None,
                      _loc_az_cache=None, carrier_map=None):
    """Two neighbours of a common cell sharing a PCI -> handover ambiguity.

    The two ambiguous cells must be on the SAME carrier (a UE can only confuse
    them if they appear on one frequency).  The common neighbour may be on any
    carrier: inter-frequency measurements are exactly how a cell reports
    neighbours on another layer.
    """
    if cell_to_sector is None: cell_to_sector = {}
    if nbr_attempts is None: nbr_attempts = {}
    if carrier_map is None: carrier_map = {}
    pm = dict(zip(df['cell_id'], df[pci_col]))
    loc_map, az_map = _loc_az_cache if _loc_az_cache else _build_loc_az_maps(df)
    rows, seen = [], set()
    for ca, na in neighbors.items():
        grp = defaultdict(list)
        for nb in na:
            pv = pm.get(nb)
            if pv is not None and not pd.isna(pv): grp[int(pv)].append(nb)
        for pv, cs in grp.items():
            if len(cs) > 1:
                for i in range(len(cs)):
                    for j in range(i+1, len(cs)):
                        if not same_carrier(carrier_map, cs[i], cs[j]):
                            continue  # ambiguity only exists within one carrier
                        co = (cell_to_sector.get(str(cs[i]), '__A') == cell_to_sector.get(str(cs[j]), '__B')
                              and cell_to_sector.get(str(cs[i])) is not None)
                        if not co:
                            co = _is_co_sector_by_id(str(cs[i]), str(cs[j]))
                        if co:
                            continue  # co-sector cells share PCI by design – skip
                        t = tuple(sorted([cs[i],cs[j]])+[ca])
                        if t not in seen:
                            seen.add(t)
                            dist, facing, _ = _cell_pair_info(cs[i], cs[j], loc_map, az_map, cell_to_sector)
                            # Attempt: max of (ca↔cs[i]) and (ca↔cs[j]) since
                            # confusion impact = handover attempt to ambiguous cell
                            p1 = tuple(sorted([ca, cs[i]]))
                            p2 = tuple(sorted([ca, cs[j]]))
                            # 0 deneme ile KAYIT YOK farkli seylerdir; ikisini
                            # de bos gostermek raporu okunamaz yapiyordu.
                            has = (p1 in nbr_attempts) or (p2 in nbr_attempts)
                            att = (max(nbr_attempts.get(p1, 0),
                                       nbr_attempts.get(p2, 0)) if has else '')
                            rows.append({'cell_1':cs[i],'cell_2':cs[j],
                                'common_neighbor':ca,'pci':pv,
                                'carrier': carrier_map.get(str(cs[i]), ''),
                                'distance_km': dist, 'facing': '✅' if facing else '❌',
                                'ho_attempts': att,
                                'type':'CONFUSION','severity':'HIGH',
                                'description':f'PCI {pv} shared, both neighbors of {ca} → handover ambiguity'})
    return enrich_df_with_sector_info(pd.DataFrame(rows))

def _mod_conflict(df, neighbors, pci_col, mod, ctype, sev, desc, cell_to_sector=None,
                  nbr_attempts=None, _loc_az_cache=None, carrier_map=None):
    if cell_to_sector is None: cell_to_sector = {}
    if nbr_attempts is None: nbr_attempts = {}
    if carrier_map is None: carrier_map = {}
    pm = dict(zip(df['cell_id'], df[pci_col]))
    loc_map, az_map = _loc_az_cache if _loc_az_cache else _build_loc_az_maps(df)
    rows, seen = [], set()
    for c, ns in neighbors.items():
        cp = pm.get(c)
        if cp is None or pd.isna(cp): continue
        for nb in ns:
            np_ = pm.get(nb)
            if np_ is None or pd.isna(np_): continue
            if not same_carrier(carrier_map, c, nb):
                continue  # different carrier -> no physical interaction
            p = tuple(sorted([c, nb]))
            if int(cp)%mod == int(np_)%mod and p not in seen:
                seen.add(p)
                dist, facing, co = _cell_pair_info(c, nb, loc_map, az_map, cell_to_sector)
                if co:
                    continue  # co-sector cells share PCI by design – skip
                att = nbr_attempts.get(p, '')
                rows.append({'cell_1':c,'cell_2':nb,'pci_1':int(cp),'pci_2':int(np_),
                    f'mod{mod}_value':int(cp)%mod,
                    'carrier': carrier_map.get(str(c), ''),
                    'distance_km': dist, 'facing': '✅' if facing else '❌',
                    'ho_attempts': att,
                    'type':ctype,'severity':sev,
                    'description':f'PCI {int(cp)} mod{mod}={int(cp)%mod} vs PCI {int(np_)} mod{mod}={int(np_)%mod} → {desc}'})
    return enrich_df_with_sector_info(pd.DataFrame(rows))

def detect_mod3_conflicts(df, nb, pci='pci', cell_to_sector=None, nbr_attempts=None, _loc_az_cache=None, carrier_map=None):
    return _mod_conflict(df, nb, pci, 3, 'MOD3_CONFLICT', 'MEDIUM', 'PSS interference', cell_to_sector, nbr_attempts, _loc_az_cache, carrier_map)
def detect_mod4_conflicts(df, nb, pci='pci', cell_to_sector=None, nbr_attempts=None, _loc_az_cache=None, carrier_map=None):
    """NR PBCH DMRS interference: the DMRS subcarrier offset is PCI mod 4
    (3GPP TS 38.211 §7.4.1.4.1)."""
    return _mod_conflict(df, nb, pci, 4, 'MOD4_CONFLICT', 'MEDIUM', 'SSB DMRS interference (NR)', cell_to_sector, nbr_attempts, _loc_az_cache, carrier_map)
def detect_mod6_conflicts(df, nb, pci='pci', cell_to_sector=None, nbr_attempts=None, _loc_az_cache=None, carrier_map=None):
    """LTE CRS frequency shift v_shift = PCI mod 6 (3GPP TS 36.211 §6.10.1.2)."""
    return _mod_conflict(df, nb, pci, 6, 'MOD6_CONFLICT', 'LOW', 'CRS frekans kayması (v_shift) çakışması', cell_to_sector, nbr_attempts, _loc_az_cache, carrier_map)
def detect_mod30_conflicts(df, nb, pci='pci', cell_to_sector=None, nbr_attempts=None, _loc_az_cache=None, carrier_map=None):
    """Uplink DM-RS / SRS base-sequence group: u depends on PCI mod 30
    (LTE TS 36.211 §5.5.1.3, NR TS 38.211 §6.3.1.1 / §6.4.1.3).
    NOT PCFICH/PHICH — those map by PCI mod 2*N_RB."""
    return _mod_conflict(df, nb, pci, 30, 'MOD30_CONFLICT', 'LOW', 'UL DM-RS / SRS dizi grubu çakışması', cell_to_sector, nbr_attempts, _loc_az_cache, carrier_map)

def detect_cosector_inconsistency(df, sector_groups, pci_col='pci', rsi_col='rsi'):
    """Co-sector cells that do NOT share a PCI (or RSI).

    Every conflict check skips co-sector pairs on the grounds that cells on one
    antenna, differing only in carrier, share a PCI by design.  Nothing ever
    verified that they actually do — so a sector whose carriers disagree slips
    through silently, and the disagreement shows up instead as a phantom
    "co-site collision" against some other sector once carriers are ignored.

    This is a configuration finding, not a 3GPP conflict: cells on different
    carriers cannot interfere. It is reported because it breaks the operator's
    own convention, and because the skip above assumes the convention holds.
    """
    rows = []
    for sec_key, members in (sector_groups or {}).items():
        members = [str(m) for m in members]
        vals = {}
        for col, name in ((pci_col, 'PCI'), (rsi_col, 'RSI')):
            if col not in df.columns:
                continue
            m = dict(zip(df['cell_id'].astype(str), df[col]))
            seen = [(c, m.get(c)) for c in members if c in m and pd.notna(m.get(c))]
            distinct = {v for _, v in seen}
            if len(distinct) > 1:
                counts = defaultdict(int)
                for _, v in seen:
                    counts[v] += 1
                majority = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                vals[name] = (seen, majority, distinct)
        if not vals:
            continue
        car = dict(zip(df['cell_id'].astype(str), df['carrier'])) \
            if 'carrier' in df.columns else {}
        for name, (seen, majority, distinct) in vals.items():
            odd = [c for c, v in seen if v != majority]
            rows.append({
                'sector': sec_key,
                'parametre': name,
                'cells': len(seen),
                'degerler': ', '.join(str(int(v)) for v in sorted(distinct)),
                'cogunluk': int(majority),
                'sapan_hucre': ', '.join(sorted(odd)),
                'sapan_tasiyici': ', '.join(sorted(
                    {str(car.get(c, '')) for c in odd if car.get(c)})),
                'type': 'COSECTOR_INCONSISTENT',
                'severity': 'MEDIUM',
                'description': f'Aynı sektördeki hücreler farklı {name} taşıyor',
            })
    return pd.DataFrame(rows)


def detect_sector_shift(df, sector_groups, cell_to_sector=None, carrier_map=None,
                        pci_col='pci'):
    """Site-level view of co-sector PCI disagreement, classified by cause.

    detect_cosector_inconsistency() reports the symptom cell by cell; this
    groups it per site and says WHY, because the causes need different actions
    and one of them means the planner is about to do the wrong thing:

      AZIMUT UYUSMAZLIGI  the carriers at this site carry different azimuths,
                          so sector grouping splits them into separate sectors.
                          Either an antenna really does point elsewhere, or one
                          layer's azimuth data is wrong.  Planning in 'sector'
                          scope will NOT merge these - they are not the same
                          sector as far as the tool can tell.
      ROTASYON            every carrier uses the same set of PCIs but hands
                          them to different sectors.  A systematic offset, not
                          noise - almost always a mis-assignment.
      FARKLI PCI KUMESI   the deviating carrier uses entirely different values.
                          Can be deliberate: a layer planned on its own.
      TEKIL SAPMA         a single cell out of step.

    Returns a DataFrame, one row per site.
    """
    if not sector_groups:
        return pd.DataFrame()
    cell_to_sector = cell_to_sector or {}
    carrier_map = carrier_map or {}
    ids = df['cell_id'].astype(str)
    pci = dict(zip(ids, df[pci_col])) if pci_col in df.columns else {}
    site = (dict(zip(ids, df['site_id'].astype(str))) if 'site_id' in df.columns
            else {c: (_extract_site_name(c) or c) for c in ids})
    az = dict(zip(ids, df['azimuth'])) if 'azimuth' in df.columns else {}

    by_site = defaultdict(list)
    for c in ids:
        by_site[site.get(c, c)].append(c)

    rows = []
    for st_name, cells in by_site.items():
        secs = {cell_to_sector.get(c) for c in cells if cell_to_sector.get(c)}
        if len(secs) < 2:
            continue
        # sector -> carrier -> PCI
        bad_secs = []
        for sec in secs:
            members = [c for c in cells if cell_to_sector.get(c) == sec]
            vals = {int(pci[c]) for c in members if c in pci and pd.notna(pci[c])}
            if len(vals) > 1:
                bad_secs.append(sec)
        if not bad_secs:
            continue

        carriers = sorted({carrier_map.get(c, CARRIER_UNKNOWN) for c in cells})
        # PCI set per carrier across the whole site
        per_car = {}
        az_per_car = {}
        for car in carriers:
            cc = [c for c in cells if carrier_map.get(c, CARRIER_UNKNOWN) == car]
            per_car[car] = tuple(sorted(int(pci[c]) for c in cc
                                        if c in pci and pd.notna(pci[c])))
            az_per_car[car] = tuple(sorted({round(float(az[c])) for c in cc
                                            if c in az and pd.notna(az[c])}))

        az_sets = {v for v in az_per_car.values() if v}
        distinct_pci_sets = {v for v in per_car.values() if v}

        # Is the PCI plan consistent when cells are grouped by the NAMING
        # convention instead of by azimuth?  If it is, the operator's own plan
        # follows the naming, and it is the azimuth column that disagrees —
        # near-conclusive evidence that the azimuth data is what is wrong,
        # not the PCI.  Worth separating, because sector grouping is built on
        # azimuth: a bad azimuth silently corrupts the sector model for the
        # whole site, not just this report.
        by_name = defaultdict(list)
        name_ok = None
        for c in cells:
            sn = _extract_sector_number(c)
            if sn is not None:
                by_name[sn].append(c)
        if by_name:
            name_ok = all(
                len({int(pci[m]) for m in mem if m in pci and pd.notna(pci[m])}) <= 1
                for mem in by_name.values())
            name_az_conflict = any(
                len({round(float(az[m])) for m in mem
                     if m in az and pd.notna(az[m])}) > 1
                for mem in by_name.values())
        else:
            name_az_conflict = False

        if name_ok and name_az_conflict:
            pattern = 'AZİMUT VERİSİ ŞÜPHELİ'
            question = ('PCI planı isimlendirmeye göre TUTARLI, azimuta göre '
                        'değil — yani plan doğru, azimut sütunu şüpheli. '
                        'Sektör grupları azimuttan kurulduğu için bu sahanın '
                        'TÜM analizi ve planı yanlış gruplama üzerine oturuyor. '
                        'Azimutu düzeltip yeniden koşun.')
        elif len(az_sets) > 1 or name_az_conflict:
            pattern = 'AZİMUT UYUŞMAZLIĞI'
            question = ('Katmanların azimutları farklı. Anten gerçekten farklı '
                        'yöne mi bakıyor, yoksa bir katmanın azimut verisi mi '
                        'yanlış? Saha kayıtlarından doğrulayın.')
        elif len(distinct_pci_sets) == 1:
            pattern = 'ROTASYON'
            question = ('Her taşıyıcı aynı PCI kümesini farklı sektörlere '
                        'vermiş — sistematik kayma. Doğru eşleme hangisi?')
        else:
            n_odd = sum(1 for sec in bad_secs)
            pattern = 'TEKİL SAPMA' if n_odd == 1 else 'FARKLI PCI KÜMESİ'
            question = ('Sapan katman ayrı mı planlanmış (kasıtlı), yoksa '
                        'hatalı mı? Kasıtlıysa plan onu birleştirecektir.')

        # which carrier deviates: the one whose per-sector PCI differs from the mode
        odd_cells = []
        for sec in bad_secs:
            members = [c for c in cells if cell_to_sector.get(c) == sec]
            counts = defaultdict(int)
            for c in members:
                if c in pci and pd.notna(pci[c]):
                    counts[int(pci[c])] += 1
            if not counts:
                continue
            majority = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            odd_cells += [c for c in members
                          if c in pci and pd.notna(pci[c]) and int(pci[c]) != majority]

        rows.append({
            'site': st_name,
            'desen': pattern,
            'sektör': len(secs),
            'taşıyıcı': len(carriers),
            'tutarsız_sektör': len(bad_secs),
            'sapan_hücre': ', '.join(sorted(odd_cells)),
            'sapan_taşıyıcı': ', '.join(sorted(
                {carrier_map.get(c, CARRIER_UNKNOWN) for c in odd_cells})),
            'hücre': len(cells),
            'isimlendirmeye_göre_PCI': ('tutarlı' if name_ok else
                                        'tutarsız' if name_ok is not None else '—'),
            'kontrol': question,
        })
    out = pd.DataFrame(rows)
    if len(out):
        order = {'AZİMUT VERİSİ ŞÜPHELİ': 0, 'AZİMUT UYUŞMAZLIĞI': 1,
                 'ROTASYON': 2, 'FARKLI PCI KÜMESİ': 3, 'TEKİL SAPMA': 4}
        out = out.sort_values(by=['desen', 'site'],
                              key=lambda col: col.map(order) if col.name == 'desen' else col)
        out = out.reset_index(drop=True)
    return out


# ============================================================
# RSI Conflict (Cell-Range-Aware)
# ============================================================
def detect_rsi_collisions(df, neighbors, rsi_col='rsi', technology='LTE', cell_to_sector=None,
                          nbr_attempts=None, _loc_az_cache=None, carrier_map=None):
    technology = norm_tech(technology)  # UI = tek otorite
    if cell_to_sector is None: cell_to_sector = {}
    if nbr_attempts is None: nbr_attempts = {}
    if carrier_map is None: carrier_map = {}
    if rsi_col not in df.columns: return pd.DataFrame()
    rm = dict(zip(df['cell_id'], df[rsi_col]))
    nm = {}   # cell_id -> Ncs
    nzm = {}  # cell_id -> Nzc (139 for short seq, 839 otherwise)
    rootm = {}  # cell_id -> roots consumed (restricted-aware)
    for _,r in df.iterrows():
        _rv = r.get(rsi_col)
        _p = _prach_params(r, technology,
                           rsi=int(_rv) if _rv is not None and not pd.isna(_rv) else None)
        nzm[r['cell_id']] = _p['nzc']
        nm[r['cell_id']] = _p['ncs']
        rootm[r['cell_id']] = (_p['roots_needed'] if _p['roots_needed']
                               else roots_needed(64, _p['ncs'] or 13, _p['nzc']))
    # Per-cell max_rsi: depends on Nzc (L=839→max 838, L=139→max 138)
    max_rsi_map = {}  # cell_id → max_rsi
    for cid, nzc in nzm.items():
        max_rsi_map[cid] = (NZC_SHORT - 1) if nzc == NZC_SHORT else (NZC_LONG - 1)
    mx_default = rsi_count(technology)
    rows, seen = [], set()
    loc_map, az_map = _loc_az_cache if _loc_az_cache else _build_loc_az_maps(df)
    for c, ns in neighbors.items():
        cr = rm.get(c)
        if cr is None or pd.isna(cr): continue
        cn = nm.get(c, 13)
        for nb in ns:
            nr_ = rm.get(nb)
            if nr_ is None or pd.isna(nr_): continue
            if not same_carrier(carrier_map, c, nb):
                continue  # PRACH is per-carrier: no cross-carrier root clash
            nn = nm.get(nb, 13)
            p = tuple(sorted([c, nb]))
            if p in seen: continue
            seen.add(p)
            co = (cell_to_sector.get(str(c), '__A') == cell_to_sector.get(str(nb), '__B')
                  and cell_to_sector.get(str(c)) is not None)
            if co:
                continue  # co-sector cells share RSI by design – skip
            c_nzc = nzm.get(c, NZC_LONG)
            nb_nzc = nzm.get(nb, NZC_LONG)
            if c_nzc != nb_nzc:
                continue  # different L_RA -> different PRACH numerology, cannot clash
            # Use the larger Nzc for overlap check (conservative)
            overlap_nzc = max(c_nzc, nb_nzc)
            # Per-cell max_rsi for wrapping
            c_mx = max_rsi_map.get(c, mx_default)
            nb_mx = max_rsi_map.get(nb, mx_default)
            pair_mx = max(c_mx, nb_mx)  # use larger for overlap check
            r1 = rootm.get(c, roots_needed(64, cn, c_nzc))
            r2 = rootm.get(nb, roots_needed(64, nn, nb_nzc))
            if rsi_overlap(int(cr), cn, int(nr_), nn, overlap_nzc, 64, pair_mx,
                           r1=r1, r2=r2):
                dist, facing, _ = _cell_pair_info(c, nb, loc_map, az_map, cell_to_sector)
                att = nbr_attempts.get(p, '')
                rows.append({
                    'cell_1':c,'cell_2':nb,'rsi_1':int(cr),'rsi_2':int(nr_),
                    'ncs_1':cn,'ncs_2':nn,'roots_cell_1':r1,'roots_cell_2':r2,
                    'carrier': carrier_map.get(str(c), ''),
                    'rsi_range_1':f'{int(cr)}-{(int(cr)+r1-1)%pair_mx}',
                    'rsi_range_2':f'{int(nr_)}-{(int(nr_)+r2-1)%pair_mx}',
                    'distance_km': dist, 'facing': '✅' if facing else '❌',
                    'ho_attempts': att,
                    'type':'RSI_COLLISION','severity':'HIGH',
                    'description':f'RSI overlap: {c} RSI {int(cr)} ({r1} roots) vs {nb} RSI {int(nr_)} ({r2} roots)'})
    return enrich_df_with_sector_info(pd.DataFrame(rows))

# ============================================================
# Full Analysis
# ============================================================
def compute_health_score(col_count, con_count, m3_count, m6_count, m30_count,
                        rsi_count, total_neighbor_pairs, n_cells=0,
                        m4_count=0, technology='LTE'):
    """Compute PCI/RSI health score 0-100.

    Scoring uses **two-tier** Mod-N penalty so that even conflict counts
    *below* the random baseline still contribute to the penalty.

    Penalty components:

    * **Collision** (W=15, sqrt) – ref = n_cells/2.  Critical: call drops.
    * **Confusion** (W=20, linear) – 20 % of pairs = full penalty.
    * **RSI**       (W=20, sqrt) – ref = n_cells * 5.  RACH failures.
    * **Mod-N**     (two-tier):
        - *Tier A — ratio penalty* (up to 25 % of Mod weight):
          Proportional to conflict_count / (tn/N).
          Even ONE conflict gives a small penalty; at random baseline
          (count ≈ tn/N) this component gives 25 % of the mod weight.
        - *Tier B — excess penalty* (up to 75 % of Mod weight):
          Only kicks in when count EXCEEDS the random baseline (tn/N).
          50 % excess above baseline → full B component.
      This design ensures score=100 requires ZERO conflicts everywhere,
      while keeping the excess-based sensitivity for truly bad planning.

    Technology-aware:
      LTE → Mod3 + Mod6 + Mod30 (PSS, RS, PCFICH)
      NR  → Mod3 + Mod4          (PSS, SSB DMRS)

    Component weights (sum to 100):
        Collision : 15
        Confusion : 20
        RSI       : 20
      LTE: Mod3=20, Mod6=15, Mod30=10
      NR:  Mod3=25, Mod4=20
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if total_neighbor_pairs <= 0:
        return 100.0
    tn = total_neighbor_pairs
    nc = max(n_cells, 1)

    W_COL, W_CON, W_RSI = 15, 20, 20

    if technology == 'NR':
        W_M3, W_M4, W_M6, W_M30 = 25, 20, 0, 0
    else:
        W_M3, W_M4, W_M6, W_M30 = 20, 0, 15, 10

    # --- Collision (sqrt): ref = n_cells / 2 ---
    ref_col = max(nc / 2.0, 1)
    p_col = min(1.0, _math.sqrt(col_count / ref_col)) * W_COL if col_count > 0 else 0

    # --- Confusion (linear): 20 % of pairs = full penalty ---
    p_con = min(1.0, con_count / (tn * 0.20)) * W_CON

    # --- RSI (sqrt): ref = n_cells * 5 ---
    ref_rsi = max(nc * 5.0, 1)
    p_rsi = min(1.0, _math.sqrt(rsi_count / ref_rsi)) * W_RSI if rsi_count > 0 else 0

    # --- Mod-N: two-tier (ratio + excess) ---
    def _mod_penalty(count, mod_n, weight):
        """Two-tier penalty for mod-N conflicts.
        Tier A (25 %): proportional to count / expected_random
        Tier B (75 %): excess above expected_random, 50 % excess = full
        """
        if weight <= 0 or count <= 0:
            return 0.0
        exp = tn / float(mod_n)
        # Tier A: ratio-based (always non-zero when count > 0)
        ratio = min(1.0, count / exp) if exp > 0 else 1.0
        tier_a = ratio * 0.25 * weight
        # Tier B: excess above random baseline
        if count > exp and exp > 0:
            excess_ratio = min(1.0, (count - exp) / (exp * 0.50))
            tier_b = excess_ratio * 0.75 * weight
        else:
            tier_b = 0.0
        return tier_a + tier_b

    p_m3 = _mod_penalty(m3_count, 3, W_M3)
    p_m4 = _mod_penalty(m4_count, 4, W_M4)
    p_m6 = _mod_penalty(m6_count, 6, W_M6)
    p_m30 = _mod_penalty(m30_count, 30, W_M30)

    pen = p_col + p_con + p_rsi + p_m3 + p_m4 + p_m6 + p_m30
    return round(max(0, 100 - pen), 2)


def _count_excl_cosector(tbl):
    """Legacy helper – co-sector rows are now skipped at detection time,
    so this simply returns len(tbl)."""
    return len(tbl)


def run_full_analysis(df, radius_km, technology='LTE', use_antenna_direction=True,
                      default_beamwidth=65.0, check_mod3=True, check_mod6=True,
                      check_mod30=None, check_rsi=True,
                      include_intra_site=True, external_neighbors=None,
                      cell_to_sector=None, progress_callback=None,
                      check_mod4=False, carrier_map=None, sector_groups=None):
    technology = norm_tech(technology)  # UI = tek otorite
    sector_groups_for_check = sector_groups or {}
    def _prog(pct, msg=''):
        if progress_callback:
            progress_callback(pct, msg)

    # Technology-aware: NR uses mod4 (SSB DMRS), LTE uses mod6/mod30
    # Auto-enable mod4 for NR, auto-disable mod6/mod30 for NR
    # NR'de mod30 ZORLA kapatilmaz: 30-gruplu low-PAPR dizi yapisi NR'de de
    # vardir, ama PCI'ya bagli olmasi kosulludur (TS 38.211 §6.3.2.2.1 —
    # hoppingId konfigure edilmisse dizi grubu PCI'dan kopar; §6.4.1.4.2 —
    # SRS zaten hep kendi sequenceId'sini kullanir).  Bu yuzden varsayilan
    # kapali ama cagiran acikca acabilir.
    if check_mod30 is None:
        check_mod30 = (technology != 'NR')
    if technology == 'NR':
        check_mod4 = True
        check_mod6 = False      # NR'de CRS yok

    _prog(0, 'Komşuluk grafiği oluşturuluyor...')
    nb, nbr_sources, nbr_attempts = find_neighbors(df, radius_km, use_antenna_direction,
                                                    default_beamwidth, include_intra_site,
                                                    external_neighbors)
    c2s = cell_to_sector or {}
    na = nbr_attempts
    # Carrier scoping (K-1): a conflict only exists between cells on the same
    # carrier.  With no carrier information every cell resolves to the same
    # bucket, so this reduces exactly to the previous behaviour.
    cm = carrier_map if carrier_map is not None else build_carrier_map(df)
    # Build location/azimuth maps ONCE for all detection functions
    _lac = _build_loc_az_maps(df)
    _prog(20, 'Collision tespiti...')
    col = detect_collisions(df, nb, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm)
    _prog(35, 'Confusion tespiti...')
    con = detect_confusions(df, nb, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm)
    _prog(50, 'Mod3 kontrolü...')
    m3 = detect_mod3_conflicts(df, nb, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm) if check_mod3 else pd.DataFrame()
    _prog(55, 'Mod4 kontrolü (NR SSB DMRS)...')
    m4 = detect_mod4_conflicts(df, nb, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm) if check_mod4 else pd.DataFrame()
    _prog(60, 'Mod6 kontrolü...')
    m6 = detect_mod6_conflicts(df, nb, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm) if check_mod6 else pd.DataFrame()
    _prog(70, 'Mod30 kontrolü...')
    m30 = detect_mod30_conflicts(df, nb, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm) if check_mod30 else pd.DataFrame()
    _prog(80, 'RSI çakışma tespiti...')
    rsi = detect_rsi_collisions(df, nb, 'rsi', technology, cell_to_sector=c2s, nbr_attempts=na, _loc_az_cache=_lac, carrier_map=cm) if check_rsi else pd.DataFrame()
    _prog(88, 'Co-sektör tutarlılığı...')
    cosec = detect_cosector_inconsistency(df, sector_groups_for_check)
    shift = detect_sector_shift(df, sector_groups_for_check, c2s, cm)
    _prog(90, 'Skor hesaplanıyor...')
    tc = len(df)
    # Neighbour pairs: the full graph, and the same-carrier subset that the
    # mod-N random baseline has to be measured against.
    tn_all = sum(len(v) for v in nb.values())//2
    tn = sum(1 for c, nbs in nb.items() for x in nbs if same_carrier(cm, c, x))//2
    n_car, n_unknown, car_counts = carrier_report(cm)
    mp = pci_count(technology)

    # Count neighbor sources
    src_counts = defaultdict(int)
    for pair, sources in nbr_sources.items():
        for s in sources:
            src_counts[s] += 1

    # ── Co-site mod3 / mod4 counting ──────────────────────────
    # A co-site pair = same physical site but different sector (not co-sector).
    # Co-sector pairs are already excluded by _mod_conflict, so any
    # remaining same-site pair in the conflict list is a true co-site conflict.
    sm = dict(zip(df['cell_id'].astype(str), df['site_id'].astype(str))) if 'site_id' in df.columns else {}
    def _count_cosite(conflict_df):
        if len(conflict_df) == 0:
            return 0
        cnt = 0
        for _, row in conflict_df.iterrows():
            c1, c2 = str(row['cell_1']), str(row['cell_2'])
            same = False
            if sm:
                same = (sm.get(c1) == sm.get(c2) and sm.get(c1) is not None
                        and sm.get(c1) != '' and sm.get(c1) != 'nan')
            if not same:
                same = _is_same_site_by_id(c1, c2)
            if same:
                cnt += 1
        return cnt
    cosite_col = _count_cosite(col)
    cosite_m3 = _count_cosite(m3)
    cosite_m4 = _count_cosite(m4)

    hs = compute_health_score(len(col), len(con), len(m3), len(m6), len(m30), len(rsi), tn,
                              n_cells=tc, m4_count=len(m4), technology=technology)

    # Per-carrier breakdown: the same formula applied to each frequency layer
    # on its own.  A blended figure hides a layer that is in trouble.
    per_carrier = []
    _car_order = sorted(k for k in car_counts if k != CARRIER_UNKNOWN)
    if CARRIER_UNKNOWN in car_counts:
        _car_order.append(CARRIER_UNKNOWN)
    for car in _car_order:
        cells_c = {cid for cid, v in cm.items() if v == car}
        pairs_c = sum(1 for c, nbs in nb.items() if str(c) in cells_c
                      for x in nbs if str(x) in cells_c)//2

        def _cnt(tbl, _cc=cells_c):
            if len(tbl) == 0:
                return 0
            return int(((tbl['cell_1'].astype(str).isin(_cc)) &
                        (tbl['cell_2'].astype(str).isin(_cc))).sum())
        c_col, c_con = _cnt(col), _cnt(con)
        c_m3, c_m4 = _cnt(m3), _cnt(m4)
        c_m6, c_m30, c_rsi = _cnt(m6), _cnt(m30), _cnt(rsi)
        per_carrier.append({
            'carrier': car, 'cells': len(cells_c), 'neighbor_pairs': pairs_c,
            'collision': c_col, 'confusion': c_con, 'mod3': c_m3, 'mod4': c_m4,
            'mod6': c_m6, 'mod30': c_m30, 'rsi': c_rsi,
            'health_score': compute_health_score(
                c_col, c_con, c_m3, c_m6, c_m30, c_rsi, pairs_c,
                n_cells=len(cells_c), m4_count=c_m4, technology=technology)})

    # Traffic exposed to conflicts — what actually turns into failed handovers.
    # Reported next to the score, never folded into it.
    _att_total = sum(nbr_attempts.values()) if nbr_attempts else 0
    _att_hard = conflicted_attempts([col, con, rsi], nbr_attempts)
    _att_all = conflicted_attempts([col, con, m3, m4, m6, m30, rsi], nbr_attempts)

    s = {'technology':technology,'total_cells':tc,'total_neighbor_pairs':tn,
         'total_neighbor_pairs_all_layers': tn_all,
         'attempts_total': _att_total,
         'attempts_on_hard_conflicts': _att_hard,
         'attempts_on_any_conflict': _att_all,
         'carrier_count': n_car, 'cells_without_carrier': n_unknown,
         'carrier_cells': car_counts, 'per_carrier': per_carrier,
         'max_pci_range':f'0-{mp-1}','search_radius_km':radius_km,
         'collision_count':len(col),'confusion_count':len(con),
         'mod3_conflict_count':len(m3),'mod4_conflict_count':len(m4),
         'mod6_conflict_count':len(m6),
         'mod30_conflict_count':len(m30),'rsi_collision_count':len(rsi),
         'cosite_collision_count': cosite_col,
         'cosite_mod3_count': cosite_m3,
         'cosite_mod4_count': cosite_m4,
         'neighbor_sources': dict(src_counts),
         'pairs_with_ho_attempts': len(nbr_attempts),
         'cosector_inconsistent_count': len(cosec),
         'sector_shift_sites': len(shift),
         'cells_with_issues':len(set(
             (col['cell_1'].tolist() if len(col)>0 else [])+(col['cell_2'].tolist() if len(col)>0 else [])+
             (con['cell_1'].tolist() if len(con)>0 else [])+(con['cell_2'].tolist() if len(con)>0 else []))),
         'health_score': hs}
    _prog(100, 'Analiz tamamlandı!')
    return {'neighbors':nb,'neighbor_sources':nbr_sources,
            'neighbor_attempts':nbr_attempts,
            'collisions':col,'confusions':con,
            'mod3_conflicts':m3,'mod4_conflicts':m4,
            'mod6_conflicts':m6,'mod30_conflicts':m30,
            'rsi_collisions':rsi,'cosector_inconsistent':cosec,
            'sector_shift':shift,
            'carrier_map':cm,'summary':s}

def build_neighbor_table(df, neighbors, nbr_sources=None, nbr_attempts=None):
    rows = []
    loc = {r['cell_id']:(r['latitude'],r['longitude']) for _,r in df.iterrows()}
    pm = dict(zip(df['cell_id'], df['pci']))
    rm = dict(zip(df['cell_id'], df['rsi'])) if 'rsi' in df.columns else {}
    sm = dict(zip(df['cell_id'], df['site_id'])) if 'site_id' in df.columns else {}
    if nbr_attempts is None:
        nbr_attempts = {}
    seen = set()
    for c, ns in neighbors.items():
        for nb in ns:
            p = tuple(sorted([c, nb]))
            if p not in seen:
                seen.add(p)
                cl, nl = loc.get(c,(0,0)), loc.get(nb,(0,0))
                d = haversine_distance(cl[0],cl[1],nl[0],nl[1])
                sources = ', '.join(sorted(nbr_sources.get(p, {'?'}))) if nbr_sources else '—'
                same_site = (sm.get(c) == sm.get(nb) and sm.get(c) is not None) if sm else False
                att = nbr_attempts.get(p)
                row = {
                    'cell_1':c, 'site_1':sm.get(c,'—'),
                    'pci_1':pm.get(c), 'rsi_1':rm.get(c),
                    'cell_2':nb, 'site_2':sm.get(nb,'—'),
                    'pci_2':pm.get(nb), 'rsi_2':rm.get(nb),
                    'distance_km':round(d,3),
                    'same_site': '✅' if same_site else '',
                    'ho_attempts': att if att is not None else '',
                    'neighbor_source': sources}
                rows.append(row)
    return enrich_df_with_sector_info(pd.DataFrame(rows))


# ============================================================
# PCI / RSI Recommendation Engine
# ============================================================
def _pci_is_clean(pci_candidate, cell_id, neighbors, pci_map,
                  check_mod3=True, check_mod6=True, check_mod30=True,
                  cell_to_sector=None, co_site_set=None, check_mod4=False):
    """Return True if pci_candidate causes NO conflict for cell_id (includes confusion)."""
    return _pci_is_clean_ex(pci_candidate, cell_id, neighbors, pci_map,
                            check_mod3, check_mod6, check_mod30, check_confusion=True,
                            cell_to_sector=cell_to_sector, co_site_set=co_site_set,
                            check_mod4=check_mod4)


def _pci_is_clean_ex(pci_candidate, cell_id, neighbors, pci_map,
                     check_mod3=True, check_mod6=True, check_mod30=True,
                     check_confusion=True, cell_to_sector=None,
                     co_site_set=None, check_mod4=False,
                     enforce_co_site_mod3=True, carrier_map=None):
    """Return True if pci_candidate causes NO conflict for cell_id among its neighbors.
       Optionally checks 2nd-ring neighbours for confusion.
       Skips co-sector neighbours (same site+azimuth).
       Co-site neighbours (same site, different sector) enforce
       collision always + mod3 when enforce_co_site_mod3=True.
    """
    if cell_to_sector is None:
        cell_to_sector = {}
    if co_site_set is None:
        co_site_set = set()
    if carrier_map is None:
        carrier_map = {}
    my_sector = cell_to_sector.get(str(cell_id))
    for nb in neighbors.get(cell_id, set()):
        # Skip co-sector neighbours
        if my_sector is not None and cell_to_sector.get(str(nb)) == my_sector:
            continue
        if not same_carrier(carrier_map, cell_id, nb):
            continue  # different carrier -> no collision / mod-N binding
        nb_pci = pci_map.get(nb)
        if nb_pci is None or pd.isna(nb_pci):
            continue
        nb_pci = int(nb_pci)
        # Collision — always checked
        if pci_candidate == nb_pci:
            return False
        # Co-site: enforce mod3 (PSS interference at same site is critical)
        is_co_site = frozenset((str(cell_id), str(nb))) in co_site_set
        if enforce_co_site_mod3 and is_co_site and pci_candidate % 3 == nb_pci % 3:
            return False
        # Normal mod-N checks (can be relaxed)
        if check_mod3 and pci_candidate % 3 == nb_pci % 3:
            return False
        if check_mod4 and pci_candidate % 4 == nb_pci % 4:
            return False
        if check_mod6 and pci_candidate % 6 == nb_pci % 6:
            return False
        if check_mod30 and pci_candidate % 30 == nb_pci % 30:
            return False
    # Confusion check
    if check_confusion:
        for nb in neighbors.get(cell_id, set()):
            if my_sector is not None and cell_to_sector.get(str(nb)) == my_sector:
                continue
            for nb2 in neighbors.get(nb, set()):
                if nb2 == cell_id:
                    continue
                # Skip if nb2 is co-sector with the candidate cell
                if my_sector is not None and cell_to_sector.get(str(nb2)) == my_sector:
                    continue
                if not same_carrier(carrier_map, cell_id, nb2):
                    continue  # ambiguity lives within one carrier
                nb2_pci = pci_map.get(nb2)
                if nb2_pci is not None and not pd.isna(nb2_pci) and int(nb2_pci) == pci_candidate:
                    return False
    return True


def _rsi_is_clean(rsi_candidate, cell_ncs, cell_id, neighbors, rsi_map, ncs_map,
                  technology='LTE', cell_to_sector=None, nzc_map=None,
                  carrier_map=None, restricted_map=None):
    """Return True if rsi_candidate causes NO RSI overlap for cell_id.

    Skips co-sector neighbours and other carriers.  Same rules as
    detect_rsi_collisions: roots are compared in the cell's own root space
    (838 for L=839, 138 for L=139 — v3 used 838 for both), a neighbour with a
    different L_RA cannot clash, and restricted cells reserve the roots of
    their actual start (restricted_map: cell -> False / 'typeA' / 'typeB').
    A candidate outside the cell's root space (>137 for L=139) is never clean.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if cell_to_sector is None:
        cell_to_sector = {}
    if nzc_map is None:
        nzc_map = {}
    if carrier_map is None:
        carrier_map = {}
    if restricted_map is None:
        restricted_map = {}
    my_sector = cell_to_sector.get(str(cell_id))
    cell_nzc = nzc_map.get(str(cell_id), NZC_LONG)
    mine = cell_root_set(rsi_candidate, cell_ncs, cell_nzc,
                         restricted_map.get(str(cell_id), False), candidate=True)
    if mine is None:
        return False
    for nb in neighbors.get(cell_id, set()):
        if my_sector is not None and cell_to_sector.get(str(nb)) == my_sector:
            continue
        if not same_carrier(carrier_map, cell_id, nb):
            continue  # PRACH is per-carrier
        nb_rsi = rsi_map.get(nb)
        if nb_rsi is None or pd.isna(nb_rsi):
            continue
        nb_nzc = nzc_map.get(str(nb), NZC_LONG)
        if nb_nzc != cell_nzc:
            continue  # different L_RA -> different root space, cannot clash
        theirs = cell_root_set(int(nb_rsi), ncs_map.get(nb, 13), nb_nzc,
                               restricted_map.get(str(nb), False))
        if theirs and (mine & theirs):
            return False
    return True


def _clean_rsi_starts(group, neighbors, rsi_map, ncs_map, nzc_map, carrier_map,
                      cell_to_sector, restricted_map, technology='LTE', exclude=None):
    """Every start that _rsi_is_clean_for_group() would accept, in one pass.

    Same rules, computed as a forbidden-start set instead of one full
    neighbour scan per candidate, so the whole root space can be offered to
    the reuse-distance choice.  Restricted groups fall back to the exact
    per-candidate check (their root count depends on the start).
    """
    nz = nzc_map.get(str(group[0]), NZC_LONG)
    mx = root_space_size(nz)
    if any(restricted_map.get(str(m), False) for m in group):
        return [c for c in range(mx) if c != exclude and _rsi_is_clean_for_group(
            c, group, neighbors, rsi_map, ncs_map, technology, cell_to_sector,
            nzc_map, carrier_map, restricted_map)]
    forbidden = set()
    for m in group:
        m = str(m)
        rn_m = cell_root_count(0, ncs_map.get(m, 13), nz, False)
        if rn_m is None or rn_m > mx:
            return []
        sec = cell_to_sector.get(m)
        for nb in neighbors.get(m, set()):
            if sec is not None and cell_to_sector.get(str(nb)) == sec:
                continue
            if not same_carrier(carrier_map, m, nb):
                continue
            v = rsi_map.get(nb)
            if v is None or pd.isna(v):
                continue
            nb_nz = nzc_map.get(str(nb), NZC_LONG)
            if nb_nz != nz:
                continue
            theirs = cell_root_set(int(v), ncs_map.get(nb, 13), nb_nz,
                                   restricted_map.get(str(nb), False))
            for r in theirs or ():
                for k in range(rn_m):
                    forbidden.add((r - k) % mx)
    return [c for c in range(mx) if c not in forbidden and c != exclude]


class _ReuseChooser:
    """Pick, among clean starts, the one whose nearest reuse is farthest away
    — the suggestion / rescan / new-cell counterpart of the planner's
    'max_reuse' strategy, evaluated against a working RSI map."""

    def __init__(self, df, ncs_map, nzc_map, restricted_map, carrier_map,
                 cell_to_sector, strategy='max_reuse'):
        ids = df['cell_id'].astype(str).tolist()
        self.lat = dict(zip(ids, pd.to_numeric(df['latitude'], errors='coerce')))
        self.lon = dict(zip(ids, pd.to_numeric(df['longitude'], errors='coerce')))
        self.ncs, self.nzc, self.rst = ncs_map, nzc_map, restricted_map
        self.car = carrier_map or {}
        self.c2s = cell_to_sector or {}
        self.strategy = strategy
        self.members = defaultdict(list)
        for c in ids:
            self.members[(self.car.get(c, CARRIER_UNKNOWN), nzc_map.get(c, NZC_LONG))].append(c)

    def _profile(self, m, rsi_map):
        nz = self.nzc.get(m, NZC_LONG)
        mx = root_space_size(nz)
        prof = np.full(mx, np.inf)
        if pd.isna(self.lat.get(m)) or pd.isna(self.lon.get(m)):
            return prof
        sec = self.c2s.get(m)
        xs, roots = [], []
        for x in self.members[(self.car.get(m, CARRIER_UNKNOWN), nz)]:
            if x == m or (sec is not None and self.c2s.get(x) == sec):
                continue
            v = rsi_map.get(x)
            if v is None or pd.isna(v) or pd.isna(self.lat.get(x)):
                continue
            s = cell_root_set(int(v), self.ncs.get(x, 13), nz, self.rst.get(x, False))
            if s:
                xs.append(x)
                roots.append(np.fromiter(s, int))
        if not xs:
            return prof
        dist = _haversine_vec(self.lat[m], self.lon[m],
                              [self.lat[x] for x in xs], [self.lon[x] for x in xs])
        np.minimum.at(prof, np.concatenate(roots),
                      np.repeat(dist, [len(r) for r in roots]))
        return prof

    def choose(self, clean, group, rsi_map):
        if not clean:
            return None
        if self.strategy == 'first_fit' or len(clean) == 1:
            return clean[0]
        group = [str(m) for m in group]
        vals = np.full(len(clean), np.inf)
        for m in group:
            prof = self._profile(m, rsi_map)
            nz = self.nzc.get(m, NZC_LONG)
            mx = len(prof)
            for j, c in enumerate(clean):
                rn = cell_root_count(c, self.ncs.get(m, 13), nz, self.rst.get(m, False))
                if rn:
                    vals[j] = min(vals[j], prof[(c + np.arange(rn)) % mx].min())
        return clean[int(np.argmax(vals))]      # ties -> lowest start


def _same_lra_group(cell_id, sector_groups, cell_to_sector, nzc_map):
    """`cell_id` plus its co-sector cells with the same L_RA — the cells that
    share one RSI.  A long and a short cell in one sector cannot share it."""
    sec = cell_to_sector.get(str(cell_id)) if cell_to_sector else None
    nz = nzc_map.get(str(cell_id), NZC_LONG)
    co = [str(c) for c in (sector_groups or {}).get(sec, []) if str(c) != str(cell_id)]
    return [str(cell_id)] + [c for c in co if nzc_map.get(c, NZC_LONG) == nz]


def _rsi_is_clean_for_group(rsi_candidate, group, neighbors, rsi_map, ncs_map,
                            technology, cell_to_sector, nzc_map, carrier_map,
                            restricted_map):
    """A sector's RSI is copied to every cell of `group`, so it must be clean
    on every one of their carriers — not just on the leader's.  v3 checked the
    leader only; on the real Samsun network that left 5 new collisions on the
    carrier the copy went to."""
    return all(
        _rsi_is_clean(rsi_candidate, ncs_map.get(m, 13), m, neighbors, rsi_map,
                      ncs_map, technology, cell_to_sector=cell_to_sector,
                      nzc_map=nzc_map, carrier_map=carrier_map,
                      restricted_map=restricted_map)
        for m in group)


def suggest_pci(df, neighbors, results, technology='LTE',
                check_mod3=True, check_mod6=True, check_mod30=None,
                sector_groups=None, cell_to_sector=None,
                nbr_attempts=None, check_mod4=False,
                progress_fn=None, carrier_map=None, planning_scope='sector'):
    """For every cell with a PCI problem, suggest a clean replacement PCI.

    Uses **pre-computed forbidden sets** for O(1) per-candidate checks
    instead of calling _pci_is_clean_ex in the inner loop.  This makes
    the function orders of magnitude faster on large networks.

    Technology-aware: NR → PCI 0-1007, mod4, no mod6/mod30.
                      LTE → PCI 0-503, mod6, mod30, no mod4.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    max_pci = pci_count(technology)

    # Technology-aware mod switching
    # NR'de mod30 ZORLA kapatilmaz: 30-gruplu low-PAPR dizi yapisi NR'de de
    # vardir, ama PCI'ya bagli olmasi kosulludur (TS 38.211 §6.3.2.2.1 —
    # hoppingId konfigure edilmisse dizi grubu PCI'dan kopar; §6.4.1.4.2 —
    # SRS zaten hep kendi sequenceId'sini kullanir).  Bu yuzden varsayilan
    # kapali ama cagiran acikca acabilir.
    if check_mod30 is None:
        check_mod30 = (technology != 'NR')
    if technology == 'NR':
        check_mod4 = True
        check_mod6 = False      # NR'de CRS yok

    pci_map = {str(k): v for k, v in zip(df['cell_id'], df['pci'])}
    str_neighbors = {str(k): {str(nb) for nb in nbs} for k, nbs in neighbors.items()}
    neighbors = str_neighbors
    if sector_groups is None: sector_groups = {}
    if cell_to_sector is None: cell_to_sector = {}
    if nbr_attempts is None: nbr_attempts = {}
    if carrier_map is None: carrier_map = build_carrier_map(df)
    if planning_scope == 'carrier':
        sector_groups, cell_to_sector = split_sector_groups_by_carrier(
            sector_groups, cell_to_sector, carrier_map)

    co_site_set = build_co_site_set(df, cell_to_sector)

    # Indoor cell set (for co-site mod3 indoor/outdoor distinction)
    _indoor_cells_sug = {str(r['cell_id']) for _, r in df.iterrows()
                         if _is_indoor_cell(str(r['cell_id']))}

    # SA-aligned weight constants
    _W_COLLISION   = 100.0
    _W_CO_SITE_M3  = 200.0   # outdoor↔outdoor
    _W_CO_SITE_M3_INDOOR = 80.0  # indoor involved
    _W_CONFUSION   = 30.0
    _W_MOD3        = 8.0 if check_mod3 else 0.0
    _W_MOD4        = 2.5 if check_mod4 else 0.0
    _W_MOD6        = 1.5 if check_mod6 else 0.0
    _W_MOD30       = 0.5 if check_mod30 else 0.0

    # Per-cell total attempt weight for prioritization
    _cell_att = defaultdict(int)
    for (ca, cb), att in nbr_attempts.items():
        _cell_att[str(ca)] += att
        _cell_att[str(cb)] += att

    # Collect cells with problems (skip co-sector pairs)
    # Use fast dict lookups instead of _cell_pair_info (no haversine needed)
    # Primary targets (must-fix): Collision, Confusion, AND co-site Mod3/Collision.
    # Plain ModN-only cells are NOT changed (too many, cosmetic issue).
    # When fixing a target cell, ALL modN constraints are also optimized.
    _primary_labels = {'Collision', 'Confusion', 'Co-site Collision', 'Co-site Mod3'}

    # Build site map for fast co-site detection
    _site_map = dict(zip(df['cell_id'].astype(str), df['site_id'].astype(str))) if 'site_id' in df.columns else {}

    # Build site → cells mapping for co-site collision hard exclusion
    _site_cells: Dict[str, list] = defaultdict(list)
    if _site_map:
        for _cid, _sid in _site_map.items():
            if _sid and _sid not in ('', 'nan'):
                _site_cells[_sid].append(_cid)
    else:
        for _cid in pci_map:
            _sn = _extract_site_name(_cid)
            if _sn:
                _site_cells[_sn].append(_cid)

    def _is_co_site(c1, c2):
        """Same physical site, different sector (not co-sector)."""
        if _site_map:
            s1v, s2v = _site_map.get(c1), _site_map.get(c2)
            if s1v and s2v and s1v == s2v and s1v not in ('', 'nan'):
                return True
        return _is_same_site_by_id(c1, c2)

    problem_cells: Dict[str, list] = defaultdict(list)
    for label, key in [('Collision','collisions'),('Confusion','confusions'),
                       ('Mod3','mod3_conflicts'),('Mod4','mod4_conflicts'),
                       ('Mod6','mod6_conflicts'),('Mod30','mod30_conflicts')]:
        tbl = results[key]
        if len(tbl) == 0:
            continue
        c1_arr = tbl['cell_1'].astype(str).values
        c2_arr = tbl['cell_2'].astype(str).values
        for c1, c2 in zip(c1_arr, c2_arr):
            # Fast co-sector check (dict + naming convention only — no haversine)
            s1 = cell_to_sector.get(c1)
            s2 = cell_to_sector.get(c2)
            if s1 is not None and s1 == s2:
                continue
            if _is_co_sector_by_id(c1, c2):
                continue
            # Upgrade label for co-site pairs (Collision/Mod3 at same site = critical)
            effective_label = label
            if label in ('Collision', 'Mod3') and _is_co_site(c1, c2):
                effective_label = f'Co-site {label}'
            problem_cells[c1].append(effective_label)
            problem_cells[c2].append(effective_label)

    # Filter: only keep cells that have at least one Collision or Confusion
    problem_cells = {cid: labels for cid, labels in problem_cells.items()
                     if any(l in _primary_labels for l in labels)}

    if not problem_cells:
        return pd.DataFrame()

    working_pci = dict(pci_map)
    suggestions = []

    severity_order = {'Co-site Collision':6,'Collision':5,'Confusion':4,
                      'Co-site Mod3':4,'Mod3':3,'Mod4':3,'Mod6':2,'Mod30':1}
    sorted_cells = sorted(problem_cells.items(),
                          key=lambda x: (-max(severity_order.get(i,0) for i in x[1]),
                                         -_cell_att.get(x[0], 0)))

    _co_sector_fixed: set = set()

    # --- Helper: build forbidden sets for cell + its co-sector cells ---
    def _build_forbidden(check_cells, wpci):
        """Pre-compute forbidden PCI values/modN-classes from neighbours.
        Returns (collision_set, confusion_set, co_site_mod3_set,
                 mod3_set, mod4_set, mod6_set, mod30_set).
        All O(1) lookups during candidate search.
        """
        check_set = set(str(c) for c in check_cells)
        col_set = set()          # PCIs causing collision
        conf_set = set()         # PCIs causing confusion (2-hop)
        cs_m3 = set()            # mod3 vals from co-site nbs (always enforced)
        m3 = set(); m4 = set(); m6 = set(); m30 = set()
        for cc in check_cells:
            my_sec = cell_to_sector.get(str(cc))
            for nb in neighbors.get(cc, set()):
                if nb in check_set:
                    continue  # co-sector cells will get same PCI
                if my_sec is not None and cell_to_sector.get(str(nb)) == my_sec:
                    continue
                npci = wpci.get(nb)
                if npci is None or pd.isna(npci):
                    continue
                npci = int(npci)
                # Collision / mod-N only bind within one carrier (K-1).
                if same_carrier(carrier_map, cc, nb):
                    col_set.add(npci)
                    if frozenset((str(cc), str(nb))) in co_site_set:
                        cs_m3.add(npci % 3)
                    m3.add(npci % 3)
                    m4.add(npci % 4)
                    m6.add(npci % 6)
                    m30.add(npci % 30)
                # 2-hop for confusion: the intermediate cell may sit on any
                # carrier (inter-frequency neighbour), but the two ambiguous
                # cells must share one.
                for nb2 in neighbors.get(nb, set()):
                    if nb2 in check_set:
                        continue
                    if my_sec is not None and cell_to_sector.get(str(nb2)) == my_sec:
                        continue
                    if not same_carrier(carrier_map, cc, nb2):
                        continue
                    n2pci = wpci.get(nb2)
                    if n2pci is not None and not pd.isna(n2pci):
                        conf_set.add(int(n2pci))
        return col_set, conf_set, cs_m3, m3, m4, m6, m30

    # --- Helper: spiral candidate order from current PCI ---
    def _spiral(cur_pci, mod3_class, _max_pci):
        start = cur_pci - (cur_pci % 3) + mod3_class
        if start < 0: start += 3
        if start >= _max_pci: start = mod3_class
        out = []
        for off in range(0, _max_pci, 3):
            up = start + off
            dn = start - off
            if up < _max_pci and up != cur_pci:
                out.append(up)
            if dn >= 0 and dn != cur_pci and dn != up:
                out.append(dn)
        return out

    # --- Relaxation configs: (mod3, mod4, mod6, mod30, confusion, cs_m3) ---
    _search_configs = [
        (check_mod3, check_mod4, check_mod6, check_mod30, True, True),
        (check_mod3, check_mod4, check_mod6, False, True, True),
        (check_mod3, check_mod4, False, False, True, True),
        (check_mod3, False, False, False, True, True),
        (False, False, False, False, True, True),
        (False, False, False, False, False, True),
        # Co-site mod3 unsatisfiable (e.g. a sector spanning carriers whose
        # neighbours use all three classes): keep confusion, drop co-site mod3.
        # Without this level the last resort dropped BOTH and created a
        # confusion that was avoidable (real Samsun network, site SM6003).
        (False, False, False, False, True, False),
        (False, False, False, False, False, False),   # last resort: drop cs_m3 too
    ]

    _total = len(sorted_cells)
    for _idx, (cell_id, issues) in enumerate(sorted_cells):
        if progress_fn and _idx % 50 == 0:
            progress_fn(_idx, _total)
        # Skip co-sector cells already fixed by their sector leader
        if cell_id in _co_sector_fixed:
            continue
        cur = working_pci.get(cell_id)
        if cur is None or pd.isna(cur):
            continue
        cur = int(cur)
        unique_issues = sorted(set(issues))

        # Co-sector cells that share this PCI
        sec_key = cell_to_sector.get(cell_id)
        co_cells = ([c for c in sector_groups.get(sec_key, []) if c != cell_id]
                     if sec_key else [])
        all_check = [cell_id] + co_cells

        # Pre-compute forbidden sets (covers cell_id + co_cells)
        (col_set, conf_set, cs_m3,
         f_m3, f_m4, f_m6, f_m30) = _build_forbidden(all_check, working_pci)

        # Co-site PCI hard exclusion (cells on same site, different sector)
        co_site_pci_fb = set()
        _my_site = _site_map.get(cell_id, '') if _site_map else ''
        if not _my_site or _my_site in ('', 'nan'):
            _sn = _extract_site_name(cell_id)
            if _sn:
                _my_site = _sn
        if _my_site and _my_site not in ('', 'nan'):
            _check_sec = set(str(c) for c in all_check)
            for _sc in _site_cells.get(_my_site, []):
                if _sc in _check_sec:
                    continue
                _sc_sec = cell_to_sector.get(_sc)
                if sec_key and _sc_sec == sec_key:
                    continue
                _sp = working_pci.get(_sc)
                if _sp is not None and not pd.isna(_sp):
                    co_site_pci_fb.add(int(_sp))

        # Already clean? (uses forbidden sets — no extra traversal)
        if (cur not in col_set and
            cur not in co_site_pci_fb and
            cur not in conf_set and
            cur % 3 not in cs_m3 and
            (not check_mod3 or cur % 3 not in f_m3) and
            (not check_mod4 or cur % 4 not in f_m4) and
            (not check_mod6 or cur % 6 not in f_m6) and
            (not check_mod30 or cur % 30 not in f_m30)):
            continue

        # Preferred mod3: classes NOT used by any 1-hop neighbor
        preferred_mod3 = [m for m in [0,1,2] if m not in f_m3]
        if not preferred_mod3:
            preferred_mod3 = [0,1,2]

        # Build spiral candidate order (skip collision PCIs entirely)
        all_cands = []
        for mod3_class in preferred_mod3 + [m for m in [0,1,2] if m not in preferred_mod3]:
            cands = _spiral(cur, mod3_class, max_pci)
            all_cands.extend(p for p in cands if p not in col_set and p not in co_site_pci_fb)

        found = None
        _found_is_clean = False  # track if found via clean path (skip score check)
        # The ladder relaxes mod30 -> mod6 -> mod4 -> mod3 BEFORE confusion.  A
        # level that only offers confusing candidates must therefore not end
        # the search: keep its best as a fallback and try relaxing mod-N first.
        # (v3 took that candidate at once — on the real Samsun network a
        # suggestion created a confusion to spare a mod4 conflict.)
        _soft_fallback = None
        for _cfg_idx, (_cm3, _cm4, _cm6, _cm30, _confuse, _cs_m3_chk) in enumerate(_search_configs):
            found_clean = None    # passes ALL checks incl. confusion
            found_soft = None     # passes modN but has confusion

            if not _confuse and _soft_fallback is not None:
                # Confusion is unavoidable (while keeping co-site mod3): take
                # the confusing candidate that kept the strictest mod-N.
                found = _soft_fallback
                break

            for cand in all_cands:
                # Co-site PCI hard exclusion
                if cand in co_site_pci_fb:
                    continue
                # Co-site mod3 (relaxable as last resort)
                if _cs_m3_chk and cand % 3 in cs_m3:
                    continue
                # ModN checks (relaxable)
                if _cm3 and cand % 3 in f_m3:
                    continue
                if _cm4 and cand % 4 in f_m4:
                    continue
                if _cm6 and cand % 6 in f_m6:
                    continue
                if _cm30 and cand % 30 in f_m30:
                    continue
                # Confusion
                if _confuse and cand in conf_set:
                    if found_soft is None:
                        found_soft = cand
                    continue
                found_clean = cand
                break

            if found_clean is not None:
                found = found_clean
                _found_is_clean = (_cfg_idx == 0)  # strictest config → guaranteed better
                break
            if found_soft is not None and _soft_fallback is None:
                _soft_fallback = found_soft
        if found is None and _soft_fallback is not None:
            found = _soft_fallback

        if found is not None:
            # If candidate is fully clean from strictest config, skip score check
            if _found_is_clean:
                # Guaranteed improvement — apply directly
                for c in all_check:
                    working_pci[c] = found
            else:
                # Safety check: only suggest if strictly better
                _check_set = set(str(c) for c in all_check)

                def _quick_score(pv):
                    s = 0.0
                    _m3 = pv % 3
                    _m4 = pv % 4
                    _m6 = pv % 6
                    _m30 = pv % 30
                    for cc in all_check:
                        my_sec = cell_to_sector.get(str(cc))
                        for nb in neighbors.get(cc, set()):
                            if nb in _check_set:
                                continue
                            if my_sec and cell_to_sector.get(str(nb)) == my_sec:
                                continue
                            npci = working_pci.get(nb)
                            if npci is None or pd.isna(npci): continue
                            # Same carrier scope as the forbidden sets (K-1):
                            # a same PCI on another carrier is not a collision.
                            if not same_carrier(carrier_map, cc, nb): continue
                            npci = int(npci)
                            if pv == npci:
                                s += _W_COLLISION
                            else:
                                nb_m3 = npci % 3
                                if _m3 == nb_m3:
                                    # Co-site mod3 (SA-aligned: outdoor 200, indoor 80)
                                    if frozenset((str(cc), str(nb))) in co_site_set:
                                        if str(cc) in _indoor_cells_sug or nb in _indoor_cells_sug:
                                            s += _W_CO_SITE_M3_INDOOR
                                        else:
                                            s += _W_CO_SITE_M3
                                    if _W_MOD3 > 0:
                                        s += _W_MOD3
                                if _W_MOD4 > 0 and _m4 == npci % 4: s += _W_MOD4
                                if _W_MOD6 > 0 and _m6 == npci % 6: s += _W_MOD6
                                if _W_MOD30 > 0 and _m30 == npci % 30: s += _W_MOD30
                        # 2-hop confusion (SA-aligned)
                        if _W_CONFUSION > 0:
                            for nb in neighbors.get(cc, set()):
                                if my_sec and cell_to_sector.get(str(nb)) == my_sec: continue
                                for nb2 in neighbors.get(nb, set()):
                                    if nb2 in _check_set: continue
                                    if my_sec and cell_to_sector.get(str(nb2)) == my_sec: continue
                                    if not same_carrier(carrier_map, cc, nb2): continue
                                    n2p = working_pci.get(nb2)
                                    if n2p is not None and not pd.isna(n2p) and int(n2p) == pv:
                                        s += _W_CONFUSION
                    # Co-site collision: hard constraint in SA → very high penalty here
                    if pv in co_site_pci_fb:
                        s += 500.0
                    return s

                old_score = _quick_score(cur)
                saved = {c: working_pci.get(c) for c in all_check}
                for c in all_check:
                    working_pci[c] = found
                new_score = _quick_score(found)

                if new_score >= old_score:
                    for c, v in saved.items():
                        if v is not None: working_pci[c] = v
                        else: working_pci.pop(c, None)
                    continue

            # Accept suggestion
            pss_new, sss_new = decompose_pci(found)
            pss_old, sss_old = decompose_pci(cur)
            suggestions.append({
                'cell_id': cell_id,
                'current_pci': cur,
                'current_pss': pss_old,
                'current_sss': sss_old,
                'suggested_pci': found,
                'suggested_pss': pss_new,
                'suggested_sss': sss_new,
                'issues': ', '.join(unique_issues),
                'reason': f"PCI {cur}→{found}: mod3 {cur%3}→{found%3}, "
                          f"mod6 {cur%6}→{found%6}, mod30 {cur%30}→{found%30}"
            })
            # Propagate same suggestion to co-sector cells
            for co_cell in co_cells:
                _co_sector_fixed.add(co_cell)
                co_cur = pci_map.get(co_cell)
                co_cur_v = int(co_cur) if co_cur is not None and not pd.isna(co_cur) else None
                co_pss_old, co_sss_old = decompose_pci(co_cur_v) if co_cur_v is not None else ('—', '—')
                co_issues = problem_cells.get(co_cell, [])
                suggestions.append({
                    'cell_id': co_cell,
                    'current_pci': co_cur_v if co_cur_v is not None else '—',
                    'current_pss': co_pss_old,
                    'current_sss': co_sss_old,
                    'suggested_pci': found,
                    'suggested_pss': pss_new,
                    'suggested_sss': sss_new,
                    'issues': ', '.join(sorted(set(co_issues))) if co_issues else ', '.join(unique_issues),
                    'reason': f"Sektör lideri {cell_id} ile aynı PCI: {found}"
                })
        else:
            suggestions.append({
                'cell_id': cell_id,
                'current_pci': cur,
                'current_pss': cur % 3,
                'current_sss': cur // 3,
                'suggested_pci': '—',
                'suggested_pss': '—',
                'suggested_sss': '—',
                'issues': ', '.join(unique_issues),
                'reason': 'Uygun PCI bulunamadı (tüm seçenekler çakışıyor)'
            })

    result_df = pd.DataFrame(suggestions)
    if not result_df.empty:
        assert_pci_range(result_df['suggested_pci'], technology, 'suggest_pci')
        for col in ['current_pci','suggested_pci','current_pss','suggested_pss','current_sss','suggested_sss']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
    result_df = enrich_df_with_sector_info(result_df)
    return result_df


# ============================================================
# NEW CELL PCI / RSI FINDER
# ============================================================
def find_optimal_pci_rsi_for_new_cells(existing_df, new_cells_df, radius_km,
                                        technology='LTE',
                                        use_antenna_direction=True,
                                        default_beamwidth=65.0,
                                        check_mod3=True, check_mod6=True,
                                        check_mod30=None,
                                        sector_groups=None, cell_to_sector=None,
                                        check_mod4=False, carrier_map=None,
                                        planning_scope='sector',
                                        rsi_strategy='max_reuse'):
    """Find optimal PCI and RSI for new cells being added to an existing network.

    existing_df: DataFrame of existing network cells (with pci, rsi, lat, lon, etc.)
    new_cells_df: DataFrame of new cells (same columns, pci/rsi can be empty)
    radius_km: search radius for neighbour discovery
    Returns a DataFrame with columns:
        cell_id, suggested_pci, pss, sss, suggested_rsi, roots_needed,
        rsi_range, neighbors_found, reason
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if sector_groups is None:
        sector_groups = {}
    if cell_to_sector is None:
        cell_to_sector = {}

    # Technology-aware mod switching
    # NR'de mod30 ZORLA kapatilmaz: 30-gruplu low-PAPR dizi yapisi NR'de de
    # vardir, ama PCI'ya bagli olmasi kosulludur (TS 38.211 §6.3.2.2.1 —
    # hoppingId konfigure edilmisse dizi grubu PCI'dan kopar; §6.4.1.4.2 —
    # SRS zaten hep kendi sequenceId'sini kullanir).  Bu yuzden varsayilan
    # kapali ama cagiran acikca acabilir.
    if check_mod30 is None:
        check_mod30 = (technology != 'NR')
    if technology == 'NR':
        check_mod4 = True
        check_mod6 = False      # NR'de CRS yok

    max_pci = pci_count(technology)
    max_rsi = rsi_count(technology)

    # Merge existing + new into one combined df for neighbour discovery
    # Mark new cells
    new_ids = set(str(c) for c in new_cells_df['cell_id'])
    combined = pd.concat([existing_df, new_cells_df], ignore_index=True)
    combined = combined.drop_duplicates(subset='cell_id', keep='last')

    # Discover neighbours for the combined network
    nb_all, _, _ = find_neighbors(combined, radius_km, use_antenna_direction,
                                  default_beamwidth, True, None)

    # Build PCI map from existing network (use original values)
    pci_map = {}
    for _, r in existing_df.iterrows():
        cid = str(r['cell_id'])
        pv = r.get('pci')
        if pv is not None and not pd.isna(pv):
            pci_map[cid] = int(pv)

    # Build RSI/Ncs maps
    rsi_map = {}
    ncs_map = {}
    nzc_map = {}
    restricted_map = {}
    for _, r in existing_df.iterrows():
        cid = str(r['cell_id'])
        rv = r.get('rsi')
        if rv is not None and not pd.isna(rv):
            rsi_map[cid] = int(rv)
        _p = _prach_params(r, technology)
        nzc_map[cid] = _p['nzc']
        ncs_map[cid] = _p['ncs']
        restricted_map[cid] = _p['restricted']
    # New cells too, up front: a new cell's same-L_RA co-sector group can
    # contain other new cells that have not been processed yet.
    for _, r in new_cells_df.iterrows():
        cid = str(r['cell_id'])
        _p = _prach_params(r, technology)
        nzc_map[cid] = _p['nzc']
        ncs_map[cid] = _p['ncs']
        restricted_map[cid] = _p['restricted']

    # Normalize neighbours to strings
    neighbors = {str(k): {str(v) for v in vs} for k, vs in nb_all.items()}
    if carrier_map is None:
        carrier_map = build_carrier_map(combined)
    if planning_scope == 'carrier':
        sector_groups, cell_to_sector = split_sector_groups_by_carrier(
            sector_groups, cell_to_sector, carrier_map)

    # Re-run sector group detection on combined network so new cells
    # get proper co-sector / co-site handling
    combined_sg, combined_c2s = detect_sector_groups(combined)
    # Merge: keep caller-supplied mappings but add any new-cell entries
    for k, v in combined_c2s.items():
        if k not in cell_to_sector:
            cell_to_sector[k] = v
    for k, v in combined_sg.items():
        if k not in sector_groups:
            sector_groups[k] = v

    # Build co-site set from combined network with updated sector info
    co_site_set = build_co_site_set(combined, cell_to_sector)
    _new_chooser = _ReuseChooser(combined, ncs_map, nzc_map, restricted_map,
                                 carrier_map, cell_to_sector, rsi_strategy)

    results = []
    # Working maps (so earlier new cell assignments propagate)
    working_pci = dict(pci_map)
    working_rsi = dict(rsi_map)
    _rsi_from_leader = set()   # new cells whose RSI came from a co-sector new cell
    _processed_new = set()     # new cells already given a PCI in this run

    for _, new_row in new_cells_df.iterrows():
        cid = str(new_row['cell_id'])
        nbs = neighbors.get(cid, set())

        # --- Compute Ncs / roots_needed for the new cell ---
        _p = _prach_params(new_row, technology)
        cell_nzc = _p['nzc']
        cell_ncs = _p['ncs']
        cell_rn = _p['roots_needed']
        nzc_map[cid] = cell_nzc
        ncs_map[cid] = cell_ncs

        nb_count = len(nbs)

        # --- PCI: find optimal clean PCI ---
        my_sector = cell_to_sector.get(cid)

        # Collect neighbour mod3 classes & used PCIs
        nb_mod3s = set()
        used_pcis = set()
        for nb in nbs:
            if my_sector is not None and cell_to_sector.get(str(nb)) == my_sector:
                continue
            npci = working_pci.get(nb)
            # Collision / mod-N bind only within the new cell's carrier (K-1)
            if npci is not None and same_carrier(carrier_map, cid, nb):
                nb_mod3s.add(npci % 3)
                used_pcis.add(npci)
            # 2-hop for confusion: intermediate cell may be on any carrier,
            # the ambiguous pair must share one
            for nb2 in neighbors.get(nb, set()):
                if nb2 == cid:
                    continue
                if my_sector is not None and cell_to_sector.get(str(nb2)) == my_sector:
                    continue
                if not same_carrier(carrier_map, cid, nb2):
                    continue
                n2pci = working_pci.get(nb2)
                if n2pci is not None:
                    used_pcis.add(n2pci)

        preferred_mod3 = [m for m in [0, 1, 2] if m not in nb_mod3s]
        if not preferred_mod3:
            preferred_mod3 = [0, 1, 2]

        # Search for PCI with relaxation
        # Tuple: (mod3, mod4, mod6, mod30, confusion, cs_m3)
        pci_configs = [
            (check_mod3, check_mod4, check_mod6, check_mod30, True, True),
            (check_mod3, check_mod4, check_mod6, False, True, True),
            (check_mod3, check_mod4, False, False, True, True),
            (check_mod3, False, False, False, True, True),
            (False, False, False, False, True, True),
            (False, False, False, False, False, True),
            # Co-site mod3 unsatisfiable (e.g. a sector spanning carriers whose
            # neighbours use all three classes): keep confusion, drop co-site mod3.
            # Without this level the last resort dropped BOTH and created a
            # confusion that was avoidable (real Samsun network, site SM6003).
            (False, False, False, False, True, False),
            (False, False, False, False, False, False),   # last resort: drop cs_m3
        ]

        def _level(cm3, cm4, cm6, cm30, confuse):
            all_mod = cm3 and confuse
            all_mod = (all_mod and cm4) if technology == 'NR' else (all_mod and cm6 and cm30)
            return ('Tam uyumlu' if all_mod
                    else 'ModN gevşetildi' if confuse else 'Collision-only')

        # A new carrier joining an EXISTING sector takes the sector's PCI when
        # that PCI is collision- and confusion-free on the new carrier: the
        # planning convention is one PCI per sector.  v3 always searched a
        # fresh PCI — on the real Samsun network 40 of 40 such cells got a
        # PCI different from their sector, which the tool then reports as a
        # co-sector inconsistency.
        _sec_members = [str(c) for c in sector_groups.get(my_sector, [])
                        if str(c) != cid] if my_sector else []
        _sec_existing = [c for c in _sec_members if c not in new_ids]
        # Existing cells plus new cells of this sector already processed
        _sec_known = [c for c in _sec_members if c not in new_ids or c in _processed_new]
        _sec_pcis = {working_pci.get(c) for c in _sec_known
                     if working_pci.get(c) is not None}
        sector_pci = next(iter(_sec_pcis)) if len(_sec_pcis) == 1 else None

        found_pci = None
        pci_level = ''
        if sector_pci is not None:
            for _cm3, _cm4, _cm6, _cm30, _confuse, _cs_m3 in pci_configs:
                if not _confuse:
                    continue        # never accept a confusion for convention's sake
                if _pci_is_clean_ex(sector_pci, cid, neighbors, working_pci,
                                    _cm3, _cm6, _cm30, _confuse,
                                    cell_to_sector=cell_to_sector,
                                    co_site_set=co_site_set,
                                    check_mod4=_cm4,
                                    enforce_co_site_mod3=_cs_m3,
                                    carrier_map=carrier_map):
                    found_pci = sector_pci
                    pci_level = 'Sektör PCI · ' + _level(_cm3, _cm4, _cm6, _cm30, _confuse)
                    break
        for _cm3, _cm4, _cm6, _cm30, _confuse, _cs_m3 in (pci_configs if found_pci is None else ()):
            for mod3_class in preferred_mod3 + [m for m in [0, 1, 2] if m not in preferred_mod3]:
                candidates = list(range(mod3_class, max_pci, 3))
                not_used = [p for p in candidates if p not in used_pcis]
                rest = [p for p in candidates if p in used_pcis]
                for pci_cand in not_used + rest:
                    if _pci_is_clean_ex(pci_cand, cid, neighbors, working_pci,
                                        _cm3, _cm6, _cm30, _confuse,
                                        cell_to_sector=cell_to_sector,
                                        co_site_set=co_site_set,
                                        check_mod4=_cm4,
                                        enforce_co_site_mod3=_cs_m3,
                                        carrier_map=carrier_map):
                        found_pci = pci_cand
                        # Determine quality level
                        all_mod = _cm3 and _confuse
                        if technology == 'NR':
                            all_mod = all_mod and _cm4
                        else:
                            all_mod = all_mod and _cm6 and _cm30
                        if all_mod:
                            pci_level = 'Tam uyumlu'
                        elif _confuse:
                            pci_level = 'ModN gevşetildi'
                        else:
                            pci_level = 'Collision-only'
                        break
                if found_pci is not None:
                    break
            if found_pci is not None:
                break

        # --- RSI: find optimal clean RSI ---
        # The group is this cell plus its same-L_RA co-sector cells that are
        # also NEW.  Existing co-sector cells are not in the output, so the
        # search must not pretend they move.
        rsi_group = [c for c in _same_lra_group(cid, sector_groups, cell_to_sector, nzc_map)
                     if c == cid or c in new_ids]
        found_rsi = None
        rsi_from_sector = False
        # Existing same-L_RA cells of the sector agree on one RSI?  Then that
        # is the sector's RSI and the new carrier takes it if it is clean there.
        _sec_rsis = {working_rsi.get(c) for c in _sec_existing
                     if working_rsi.get(c) is not None
                     and nzc_map.get(c, NZC_LONG) == cell_nzc}
        sector_rsi = next(iter(_sec_rsis)) if len(_sec_rsis) == 1 else None
        if cid in _rsi_from_leader:
            # Already given by an earlier new cell of the same sector, and
            # checked clean for this cell then.
            found_rsi = working_rsi.get(cid)
        elif sector_rsi is not None and _rsi_is_clean_for_group(
                int(sector_rsi), rsi_group, neighbors, working_rsi, ncs_map,
                technology, cell_to_sector, nzc_map, carrier_map, restricted_map):
            found_rsi = int(sector_rsi)
            rsi_from_sector = True
        else:
            clean = _clean_rsi_starts(rsi_group, neighbors, working_rsi, ncs_map,
                                      nzc_map, carrier_map, cell_to_sector,
                                      restricted_map, technology)
            found_rsi = _new_chooser.choose(clean, rsi_group, working_rsi)

        # Store results
        pss, sss = decompose_pci(found_pci) if found_pci is not None else ('—', '—')
        if found_rsi is not None:
            cell_rn = cell_root_count(found_rsi, cell_ncs, cell_nzc,
                                      restricted_map.get(cid, False))
        rsi_range = (f"{found_rsi}-{(found_rsi + cell_rn - 1) % root_space_size(cell_nzc)}"
                     if found_rsi is not None else '—')

        results.append({
            'cell_id': cid,
            'suggested_pci': found_pci if found_pci is not None else '—',
            'pss': pss,
            'sss': sss,
            'suggested_rsi': found_rsi if found_rsi is not None else '—',
            'ncs': cell_ncs,
            'roots_needed': cell_rn,
            'rsi_range': rsi_range,
            'neighbors_found': nb_count,
            'carrier': carrier_map.get(cid, CARRIER_UNKNOWN),
            'same_carrier_neighbors': sum(
                1 for _n in nbs if same_carrier(carrier_map, cid, _n)),
            'pci_quality': pci_level if found_pci is not None else 'Bulunamadı',
            'reason': (f"PCI={found_pci} (mod3={found_pci%3}) [{pci_level}], RSI={found_rsi}"
                       + (" [Sektör RSI]" if rsi_from_sector else "")
                       if found_pci is not None
                       else 'Uygun PCI/RSI bulunamadı')
        })

        # Propagate to working maps for next new cell
        if found_pci is not None:
            # Only this cell: existing co-sector cells keep their PCI, and a
            # later new cell of the same sector picks this one up as the
            # sector PCI (_sec_known) and checks it on its own carrier.
            working_pci[cid] = found_pci
        _processed_new.add(cid)
        if found_rsi is not None:
            working_rsi[cid] = found_rsi
            # Same RSI for the new same-L_RA co-sector cells (checked clean above)
            for co_cell in rsi_group[1:]:
                if co_cell not in _rsi_from_leader:
                    working_rsi[co_cell] = found_rsi
                    _rsi_from_leader.add(co_cell)

    _out = pd.DataFrame(results)
    if not _out.empty and 'suggested_pci' in _out.columns:
        assert_pci_range(_out['suggested_pci'], technology,
                         'find_optimal_pci_rsi_for_new_cells')
    return enrich_df_with_sector_info(_out)


# ============================================================
# PER-CELL / PER-SITE PCI & RSI RESCAN
# ============================================================
def rescan_pci_rsi_for_cells(df, neighbors, target_cell_ids,
                              technology='LTE',
                              check_mod3=True, check_mod6=True,
                              check_mod30=None, check_mod4=False,
                              sector_groups=None, cell_to_sector=None,
                              rescan_pci=True, rescan_rsi=True,
                              carrier_map=None, planning_scope='sector',
                              rsi_strategy='max_reuse'):
    """Re-optimise PCI and/or RSI **only** for *target_cell_ids* while keeping
    the rest of the network fixed.

    Uses pre-computed forbidden sets for fast O(1) candidate checks.

    Technology-aware: NR → PCI 0-1007, mod4, no mod6/mod30.
                      LTE → PCI 0-503, mod6, mod30, no mod4.
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if sector_groups is None: sector_groups = {}
    if cell_to_sector is None: cell_to_sector = {}
    if carrier_map is None: carrier_map = build_carrier_map(df)
    if planning_scope == 'carrier':
        sector_groups, cell_to_sector = split_sector_groups_by_carrier(
            sector_groups, cell_to_sector, carrier_map)

    # Technology-aware mod switching
    # NR'de mod30 ZORLA kapatilmaz: 30-gruplu low-PAPR dizi yapisi NR'de de
    # vardir, ama PCI'ya bagli olmasi kosulludur (TS 38.211 §6.3.2.2.1 —
    # hoppingId konfigure edilmisse dizi grubu PCI'dan kopar; §6.4.1.4.2 —
    # SRS zaten hep kendi sequenceId'sini kullanir).  Bu yuzden varsayilan
    # kapali ama cagiran acikca acabilir.
    if check_mod30 is None:
        check_mod30 = (technology != 'NR')
    if technology == 'NR':
        check_mod4 = True
        check_mod6 = False      # NR'de CRS yok

    max_pci = pci_count(technology)
    max_rsi = rsi_count(technology)

    target_set = set(str(c) for c in target_cell_ids)

    # Build PCI / RSI / Ncs maps from full network
    pci_map = {}
    rsi_map = {}
    ncs_map = {}
    nzc_map = {}
    restricted_map = {}
    for _, r in df.iterrows():
        cid = str(r['cell_id'])
        pv = r.get('pci')
        if pv is not None and not pd.isna(pv):
            pci_map[cid] = int(pv)
        rv = r.get('rsi')
        if rv is not None and not pd.isna(rv):
            rsi_map[cid] = int(rv)
        _p = _prach_params(r, technology)
        nzc_map[cid] = _p['nzc']
        ncs_map[cid] = _p['ncs']
        restricted_map[cid] = _p['restricted']

    # Normalize neighbours to strings
    str_neighbors = {str(k): {str(v) for v in vs} for k, vs in neighbors.items()}
    _rescan_chooser = _ReuseChooser(df, ncs_map, nzc_map, restricted_map, carrier_map,
                                    cell_to_sector, rsi_strategy)

    # Build co-site set
    co_site_set = build_co_site_set(df, cell_to_sector)

    # Build site → cells mapping for co-site collision hard exclusion
    _site_map_r: Dict[str, str] = {}
    _site_cells_r: Dict[str, list] = defaultdict(list)
    if 'site_id' in df.columns:
        for _, _r in df.iterrows():
            _cid = str(_r['cell_id'])
            _sid = str(_r['site_id'])
            _site_map_r[_cid] = _sid
            if _sid and _sid not in ('', 'nan'):
                _site_cells_r[_sid].append(_cid)
    else:
        for _cid in pci_map:
            _sn = _extract_site_name(_cid)
            if _sn:
                _site_map_r[_cid] = _sn
                _site_cells_r[_sn].append(_cid)

    # Working maps — start from current values
    working_pci = dict(pci_map)
    working_rsi = dict(rsi_map)

    # --- Forbidden-set builder (same as suggest_pci) ---
    def _build_forbidden(check_cells, wpci):
        check_set = set(str(c) for c in check_cells)
        col_set = set(); conf_set = set(); cs_m3 = set()
        m3 = set(); m4 = set(); m6 = set(); m30 = set()
        for cc in check_cells:
            my_sec = cell_to_sector.get(str(cc))
            for nb in str_neighbors.get(cc, set()):
                if nb in check_set: continue
                if my_sec is not None and cell_to_sector.get(str(nb)) == my_sec: continue
                npci = wpci.get(nb)
                if npci is None or pd.isna(npci): continue
                npci = int(npci)
                if same_carrier(carrier_map, cc, nb):   # K-1
                    col_set.add(npci)
                    if frozenset((str(cc), str(nb))) in co_site_set:
                        cs_m3.add(npci % 3)
                    m3.add(npci % 3); m4.add(npci % 4)
                    m6.add(npci % 6); m30.add(npci % 30)
                for nb2 in str_neighbors.get(nb, set()):
                    if nb2 in check_set: continue
                    if my_sec is not None and cell_to_sector.get(str(nb2)) == my_sec: continue
                    if not same_carrier(carrier_map, cc, nb2): continue
                    n2pci = wpci.get(nb2)
                    if n2pci is not None and not pd.isna(n2pci):
                        conf_set.add(int(n2pci))
        return col_set, conf_set, cs_m3, m3, m4, m6, m30

    # Relaxation configs: (mod3, mod4, mod6, mod30, confusion, cs_m3)
    pci_configs = [
        (check_mod3, check_mod4, check_mod6, check_mod30, True, True),
        (check_mod3, check_mod4, check_mod6, False, True, True),
        (check_mod3, check_mod4, False, False, True, True),
        (check_mod3, False, False, False, True, True),
        (False, False, False, False, True, True),
        (False, False, False, False, False, True),
        # Co-site mod3 unsatisfiable (e.g. a sector spanning carriers whose
        # neighbours use all three classes): keep confusion, drop co-site mod3.
        # Without this level the last resort dropped BOTH and created a
        # confusion that was avoidable (real Samsun network, site SM6003).
        (False, False, False, False, True, False),
        (False, False, False, False, False, False),   # last resort: drop cs_m3 too
    ]

    results_list = []

    for cid in target_set:
        cur_pci = pci_map.get(cid)
        cur_rsi = rsi_map.get(cid)
        cell_ncs = ncs_map.get(cid, 13)
        cell_nzc = nzc_map.get(cid, NZC_LONG)
        cell_rn = roots_needed(64, cell_ncs, cell_nzc)
        nbs = str_neighbors.get(cid, set())
        my_sector = cell_to_sector.get(cid)

        # Co-sector cells
        co_cells = ([c for c in sector_groups.get(my_sector, []) if c != cid]
                     if my_sector else [])
        all_check = [cid] + co_cells

        # --- PCI RESCAN ---
        found_pci = cur_pci
        pci_level = '— Aynı'
        pci_changed = False
        if rescan_pci:
            # Temporarily remove this cell + co-sector from working map
            saved_pci = working_pci.pop(cid, None)
            saved_co = {}
            for co_cell in co_cells:
                if co_cell in working_pci:
                    saved_co[co_cell] = working_pci.pop(co_cell)

            # Build forbidden sets
            (col_set, conf_set, cs_m3,
             f_m3, f_m4, f_m6, f_m30) = _build_forbidden(all_check, working_pci)

            # Co-site PCI hard exclusion
            co_site_pci_fb = set()
            _my_site_r = _site_map_r.get(cid, '')
            if not _my_site_r or _my_site_r in ('', 'nan'):
                _sn = _extract_site_name(cid)
                if _sn:
                    _my_site_r = _sn
            if _my_site_r and _my_site_r not in ('', 'nan'):
                _check_sec = set(str(c) for c in all_check)
                for _sc in _site_cells_r.get(_my_site_r, []):
                    if _sc in _check_sec:
                        continue
                    _sc_sec = cell_to_sector.get(_sc)
                    if my_sector and _sc_sec == my_sector:
                        continue
                    _sp = working_pci.get(_sc)
                    if _sp is not None and not pd.isna(_sp):
                        co_site_pci_fb.add(int(_sp))

            preferred_mod3 = [m for m in [0,1,2] if m not in f_m3]
            if not preferred_mod3: preferred_mod3 = [0,1,2]

            # Build spiral candidate order
            base_pci = saved_pci if saved_pci is not None else 0
            all_cands = []
            for mod3_class in preferred_mod3 + [m for m in [0,1,2] if m not in preferred_mod3]:
                start = base_pci - (base_pci % 3) + mod3_class
                if start < 0: start += 3
                if start >= max_pci: start = mod3_class
                cands = []
                for off in range(0, max_pci, 3):
                    up = start + off
                    dn = start - off
                    if up < max_pci: cands.append(up)
                    if dn >= 0 and dn != up: cands.append(dn)
                not_col = [p for p in cands if p not in col_set and p not in co_site_pci_fb]
                in_col = [p for p in cands if p in col_set or p in co_site_pci_fb]
                all_cands.extend(not_col + in_col)

            found_pci = None
            pci_level = ''
            # Same ladder rule as suggest_pci: a confusing candidate is only a
            # fallback until mod-N has been relaxed.  v3 took it at the first
            # level AND labelled it from the level's flags, so a PCI that
            # created a confusion could be reported as 'Tam uyumlu'.
            _soft_fallback = None
            for _cm3, _cm4, _cm6, _cm30, _confuse, _cs_m3_chk in pci_configs:
                if not _confuse and _soft_fallback is not None:
                    found_pci = _soft_fallback
                    pci_level = 'Confusion kabul edildi'
                    break
                found_clean = None
                found_soft = None
                for cand in all_cands:
                    if cand in col_set: continue
                    if cand in co_site_pci_fb: continue
                    if _cs_m3_chk and cand % 3 in cs_m3: continue
                    if _cm3 and cand % 3 in f_m3: continue
                    if _cm4 and cand % 4 in f_m4: continue
                    if _cm6 and cand % 6 in f_m6: continue
                    if _cm30 and cand % 30 in f_m30: continue
                    if _confuse and cand in conf_set:
                        if found_soft is None: found_soft = cand
                        continue
                    found_clean = cand
                    break
                if found_clean is not None:
                    found_pci = found_clean
                    all_mod = _cm3 and _confuse
                    if technology == 'NR': all_mod = all_mod and _cm4
                    else: all_mod = all_mod and _cm6 and _cm30
                    if all_mod: pci_level = 'Tam uyumlu'
                    elif _confuse: pci_level = 'ModN gevşetildi'
                    else: pci_level = 'Collision-only'
                    break
                if found_soft is not None and _soft_fallback is None:
                    _soft_fallback = found_soft
            if found_pci is None and _soft_fallback is not None:
                found_pci = _soft_fallback
                pci_level = 'Confusion kabul edildi'

            if found_pci is None:
                found_pci = saved_pci
                pci_level = 'Bulunamadı (mevcut korundu)'
            else:
                pci_changed = (found_pci != saved_pci)

            # Restore co-sector cells then apply update
            for co_cell, co_val in saved_co.items():
                working_pci[co_cell] = co_val
            if found_pci is not None:
                working_pci[cid] = found_pci
                for co_cell in co_cells:
                    working_pci[co_cell] = found_pci

        # --- RSI RESCAN ---
        found_rsi = cur_rsi
        rsi_changed = False
        rsi_info = '—'
        if rescan_rsi and 'rsi' in df.columns:
            saved_rsi = working_rsi.pop(cid, None)
            saved_co_rsi = {}
            if my_sector:
                for co_cell in sector_groups.get(my_sector, []):
                    if co_cell != cid and co_cell in working_rsi:
                        saved_co_rsi[co_cell] = working_rsi.pop(co_cell)

            # Same RSI goes to the same-L_RA co-sector cells, so it has to be
            # clean on all of their carriers; stay inside this cell's root space.
            rsi_group = _same_lra_group(cid, sector_groups, cell_to_sector, nzc_map)
            clean = _clean_rsi_starts(rsi_group, str_neighbors, working_rsi, ncs_map,
                                      nzc_map, carrier_map, cell_to_sector,
                                      restricted_map, technology)
            found_rsi = _rescan_chooser.choose(clean, rsi_group, working_rsi)

            if found_rsi is None:
                found_rsi = saved_rsi
                rsi_info = 'Bulunamadı (mevcut korundu)'
            else:
                rsi_changed = (found_rsi != saved_rsi)
                rn = cell_root_count(found_rsi, cell_ncs, cell_nzc,
                                     restricted_map.get(cid, False))
                rsi_info = f"RSI {found_rsi}, {rn} root"

            for co_cell, co_val in saved_co_rsi.items():
                working_rsi[co_cell] = co_val
            if found_rsi is not None:
                working_rsi[cid] = found_rsi
                for co_cell in rsi_group[1:]:
                    working_rsi[co_cell] = found_rsi

        pss, sss = decompose_pci(found_pci) if found_pci is not None else ('—', '—')
        results_list.append({
            'cell_id': cid,
            'current_pci': cur_pci if cur_pci is not None else '—',
            'suggested_pci': found_pci if found_pci is not None else '—',
            'pss': pss, 'sss': sss,
            'pci_quality': pci_level,
            'current_rsi': cur_rsi if cur_rsi is not None else '—',
            'suggested_rsi': found_rsi if found_rsi is not None else '—',
            'rsi_info': rsi_info,
            'neighbors': len(nbs),
            'changed': '✅' if (pci_changed or rsi_changed) else '—',
        })

    result_df = pd.DataFrame(results_list)
    if not result_df.empty and 'suggested_pci' in result_df.columns:
        assert_pci_range(result_df['suggested_pci'], technology,
                         'rescan_pci_rsi_for_cells')
    result_df = enrich_df_with_sector_info(result_df)
    return result_df


def suggest_rsi(df, neighbors, results, technology='LTE',
               sector_groups=None, cell_to_sector=None,
               progress_fn=None, carrier_map=None, planning_scope='sector',
               rsi_strategy='max_reuse'):
    """For every cell with an RSI problem, suggest a clean replacement RSI.

    If sector_groups is provided, co-sector cells (same site + same azimuth)
    are assigned the SAME RSI.

    Returns a list of dicts:
        cell_id, current_rsi, suggested_rsi, ncs, roots_needed, reason
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if sector_groups is None:
        sector_groups = {}
    if cell_to_sector is None:
        cell_to_sector = {}
    if carrier_map is None:
        carrier_map = build_carrier_map(df)
    if planning_scope == 'carrier':
        sector_groups, cell_to_sector = split_sector_groups_by_carrier(
            sector_groups, cell_to_sector, carrier_map)

    rsi_tbl = results.get('rsi_collisions', pd.DataFrame())
    if len(rsi_tbl) == 0:
        return pd.DataFrame()

    max_rsi = rsi_count(technology)
    rsi_map = {str(k): v for k, v in zip(df['cell_id'], df['rsi'])} if 'rsi' in df.columns else {}
    # Normalize neighbor keys to strings
    str_neighbors = {str(k): {str(nb) for nb in nbs} for k, nbs in neighbors.items()}
    neighbors = str_neighbors
    ncs_map = {}
    nzc_map = {}
    tseq_map = {}
    restricted_map = {}
    for _, r in df.iterrows():
        cid = str(r['cell_id'])
        _p = _prach_params(r, technology)
        nzc_map[cid] = _p['nzc']
        ncs_map[cid] = _p['ncs']
        tseq_map[cid] = _p['tseq_us']
        restricted_map[cid] = _p['restricted']
    chooser = _ReuseChooser(df, ncs_map, nzc_map, restricted_map, carrier_map,
                            cell_to_sector, rsi_strategy)

    def _rng(start, cid):
        """'a-b' root range of `cid` from `start`, wrapped in its own root space."""
        rn = cell_root_count(start, ncs_map.get(cid, 13), nzc_map.get(cid, NZC_LONG),
                             restricted_map.get(cid, False))
        mx = root_space_size(nzc_map.get(cid, NZC_LONG))
        return rn, f"{start}-{(start + (rn or 1) - 1) % mx}"

    # Collect problem cells
    problem_cells: Dict[str, list] = defaultdict(list)
    for _, row in rsi_tbl.iterrows():
        problem_cells[str(row['cell_1'])].append(str(row['cell_2']))
        problem_cells[str(row['cell_2'])].append(str(row['cell_1']))

    working_rsi = dict(rsi_map)
    suggestions = []
    _rsi_co_sector_fixed: set = set()
    _rsi_items = list(problem_cells.items())
    _rsi_total = len(_rsi_items)

    for _ridx, (cell_id, conflicting) in enumerate(_rsi_items):
        if progress_fn and _ridx % 50 == 0:
            progress_fn(_ridx, _rsi_total)
        # Skip co-sector cells already fixed by their sector leader
        if cell_id in _rsi_co_sector_fixed:
            continue
        cur = working_rsi.get(cell_id)
        if cur is None or pd.isna(cur):
            continue
        cur = int(cur)
        cell_ncs = ncs_map.get(cell_id, 13)
        cell_nzc = nzc_map.get(cell_id, NZC_LONG)
        rn, cur_range = _rng(cur, cell_id)

        # Already clean after earlier fix?
        if _rsi_is_clean(cur, cell_ncs, cell_id, neighbors, working_rsi, ncs_map, technology,
                         cell_to_sector=cell_to_sector, nzc_map=nzc_map,
                         carrier_map=carrier_map, restricted_map=restricted_map):
            continue

        # The RSI goes to every same-L_RA cell of the sector, so it must be
        # clean for all of them.  Candidates stay inside this cell's root
        # space (0-137 for L=139).
        group = [c for c in _same_lra_group(cell_id, sector_groups, cell_to_sector, nzc_map)
                 if c == cell_id or c in nzc_map]
        clean = _clean_rsi_starts(group, neighbors, working_rsi, ncs_map, nzc_map,
                                  carrier_map, cell_to_sector, restricted_map,
                                  technology, exclude=cur)
        found = chooser.choose(clean, group, working_rsi)

        if found is not None:
            new_rn, new_range = _rng(found, cell_id)
            _tseq = tseq_map.get(cell_id, 800.0)
            _cr = round(cell_range_from_ncs(cell_ncs, cell_nzc, _tseq), 2)
            suggestions.append({
                'cell_id': cell_id,
                'current_rsi': cur,
                'suggested_rsi': found,
                'ncs': cell_ncs,
                'roots_needed': new_rn,
                'cell_range_km': _cr,
                'current_root_range': cur_range,
                'suggested_root_range': new_range,
                'conflicting_with': ', '.join(conflicting),
                'reason': f"RSI {cur}→{found}: root aralığı çakışma yok"
            })
            working_rsi[cell_id] = found  # propagate
            # Propagate the same RSI to the same-L_RA co-sector cells
            for co_cell in group[1:]:
                working_rsi[co_cell] = found
                _rsi_co_sector_fixed.add(co_cell)
                # Add suggestion row for co-sector cell
                co_cur_rsi = rsi_map.get(co_cell)
                co_cur_v = int(co_cur_rsi) if co_cur_rsi is not None and not pd.isna(co_cur_rsi) else None
                co_ncs = ncs_map.get(co_cell, 13)
                co_nzc = nzc_map.get(co_cell, NZC_LONG)
                co_rn, co_new_range = _rng(found, co_cell)
                co_tseq = tseq_map.get(co_cell, 800.0)
                co_cr = round(cell_range_from_ncs(co_ncs, co_nzc, co_tseq), 2)
                suggestions.append({
                    'cell_id': co_cell,
                    'current_rsi': co_cur_v if co_cur_v is not None else '—',
                    'suggested_rsi': found,
                    'ncs': co_ncs,
                    'roots_needed': co_rn,
                    'cell_range_km': co_cr,
                    'current_root_range': _rng(co_cur_v, co_cell)[1] if co_cur_v is not None else '—',
                    'suggested_root_range': co_new_range,
                    'conflicting_with': ', '.join(problem_cells.get(co_cell, conflicting)),
                    'reason': f"Sektör lideri {cell_id} ile aynı RSI: {found}"
                })
        else:
            _tseq = tseq_map.get(cell_id, 800.0)
            _cr = round(cell_range_from_ncs(cell_ncs, cell_nzc, _tseq), 2)
            suggestions.append({
                'cell_id': cell_id,
                'current_rsi': cur,
                'suggested_rsi': '—',
                'ncs': cell_ncs,
                'roots_needed': rn,
                'cell_range_km': _cr,
                'current_root_range': cur_range,
                'suggested_root_range': '—',
                'conflicting_with': ', '.join(conflicting),
                'reason': 'Uygun RSI bulunamadı'
            })

    result_df = pd.DataFrame(suggestions)
    if not result_df.empty:
        for col in ['current_rsi','suggested_rsi','roots_needed','ncs']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
    result_df = enrich_df_with_sector_info(result_df)
    return result_df


# ============================================================
# Full Network RSI Auto-Planner (Cell-Range-Aware)
# ============================================================
RSI_STRATEGIES = ('max_reuse', 'first_fit')


def plan_rsi_network(df, neighbors, technology='LTE',
                     sector_groups=None, cell_to_sector=None,
                     progress_callback=None,
                     carrier_map=None, planning_scope='sector',
                     rsi_strategy='max_reuse', _priority_cells=None):
    """Assign RSI values to ALL cells from scratch using cell-range-aware planning.

    rsi_strategy — which clean start a cell gets:
      'max_reuse'  the one whose nearest reuse (same carrier, same L_RA, any
                   shared root) is FARTHEST away; unused roots count as
                   infinitely far, so the whole root space is used before any
                   root is reused.  (Ö-2)
      'first_fit'  the lowest clean start — v3 before 2026-09.  Collision-free
                   too, but it packs cells into the low roots and reuses them
                   as close as the constraints allow.

    If sector_groups is provided, co-sector cells (same site + same azimuth)
    are assigned the SAME RSI.

    Algorithm:
    1. For each cell, compute roots_needed from its Ncs (zeroCorrelationZoneConfig).
    2. Build a neighbor graph. Two cells that are neighbors must NOT have
       overlapping root-sequence index ranges.
    3. Use a greedy graph-coloring approach: process cells ordered by
       (highest roots_needed first, then highest degree) so the hardest
       cells get first pick.
    4. For each cell, pick the lowest RSI whose [RSI, RSI+roots_needed)
       interval doesn't overlap with any already-assigned neighbor interval.
    5. Co-sector cells get the same RSI.

    Returns DataFrame with:
        cell_id, current_rsi, planned_rsi, ncs, roots_needed,
        planned_range, cell_range_km, reuse_km, reuse_with, reason
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if rsi_strategy not in RSI_STRATEGIES:
        raise ValueError(f"rsi_strategy {rsi_strategy!r} — beklenen: {RSI_STRATEGIES}")
    _call_args = (df, neighbors, technology, sector_groups, cell_to_sector,
                  progress_callback, carrier_map, planning_scope)
    if sector_groups is None:
        sector_groups = {}
    if cell_to_sector is None:
        cell_to_sector = {}
    if carrier_map is None:
        carrier_map = build_carrier_map(df)
    if planning_scope == 'carrier':
        sector_groups, cell_to_sector = split_sector_groups_by_carrier(
            sector_groups, cell_to_sector, carrier_map)
    # PRACH lives on one carrier: root sequences of cells on different
    # carriers can never overlap, so the graph is scoped unconditionally.
    neighbors = scope_neighbors_by_carrier(neighbors, carrier_map)

    max_rsi = rsi_count(technology)

    # Build per-cell info
    cell_info = {}
    for _, row in df.iterrows():
        cid = str(row['cell_id'])
        _cur_rsi_raw = row.get('rsi')
        _p = _prach_params(row, technology,
                           rsi=int(_cur_rsi_raw) if _cur_rsi_raw is not None
                           and not pd.isna(_cur_rsi_raw) else None)
        is_short = _p['is_short']
        nzc = _p['nzc']
        ncs_raw = _p['ncs']
        ncs = ncs_raw if ncs_raw else 13  # display fallback; Ncs=0 -> 64 roots below
        tseq = _p['tseq_us']
        # Unrestricted: the root count is a constant of the cell.  Restricted
        # (high-speed): it depends on the START index, so this is only the
        # count at the current RSI and is used for ordering; what a cell
        # actually reserves is computed at each candidate start by _rn_at()
        # (K-2).  v3 used to reserve the current-RSI count at the new RSI and
        # left collisions in its own plan.
        rn = _p['roots_needed'] or roots_needed(64, ncs, nzc)
        cr_km = cell_range_from_ncs(ncs, nzc, tseq)
        # Number of logical root indices = wrap modulus: 838 (L=839) / 138 (L=139)
        cell_max_rsi = (NZC_SHORT - 1) if is_short else max_rsi
        cur_rsi = row.get('rsi')
        _rs = _p['restricted']
        cell_info[cid] = {
            'ncs': ncs, 'ncs_raw': ncs_raw, 'nzc': nzc, 'roots_needed': rn,
            'cell_range_km': round(cr_km, 2),
            'current_rsi': int(cur_rsi) if pd.notna(cur_rsi) else None,
            'degree': len(neighbors.get(cid, set())),
            'max_rsi': cell_max_rsi,
            'restricted': bool(_rs),
            'set_type': 'B' if _rs and str(_rs).upper().endswith('B') else 'A',
        }

    # Sort: highest roots_needed first, then highest neighbor-degree.
    # _priority_cells (internal): cells a previous max_reuse pass could not
    # place go first, before the root space is spread thin.
    _prio = set(_priority_cells or ())
    sorted_cells = sorted(cell_info.keys(),
                          key=lambda c: (c not in _prio,
                                         -cell_info[c]['roots_needed'],
                                         -cell_info[c]['degree']))

    assigned = {}      # cell_id -> start RSI
    assigned_rn = {}   # cell_id -> roots consumed FROM that start

    # Build reverse sector lookup: cell → list of co-sector cells
    _sector_members_rsi = {}  # cell_id → [all cells in same sector]
    for sk, members in sector_groups.items():
        for m in members:
            _sector_members_rsi[m] = list(members)

    def _rsi_group(cid):
        """Cells that take the same RSI as `cid`: its co-sector cells with the
        same L_RA.  A long (L=839) and a short (L=139) cell in one sector live
        in different root spaces (0-837 vs 0-137) and cannot share an RSI —
        v3 used to copy a long cell's RSI onto its short co-sector cell and
        wrote values above 137."""
        nz = cell_info[cid]['nzc']
        return [m for m in _sector_members_rsi.get(cid, [cid])
                if m in cell_info and cell_info[m]['nzc'] == nz] or [cid]

    _rn_cache = {}

    def _rn_at(cid, start):
        """Roots `cid` consumes when it starts at `start` (None = cannot reach 64)."""
        info = cell_info[cid]
        if not info['restricted']:
            return info['roots_needed']
        key = (info['ncs_raw'], info['nzc'], info['set_type'], start)
        if key not in _rn_cache:
            _rn_cache[key] = roots_needed_for_cell(
                start, info['ncs_raw'], info['nzc'], restricted=True,
                set_type=info['set_type'])
        return _rn_cache[key]

    def _group_rn_at(group, start):
        best = 0
        for m in group:
            r = _rn_at(m, start)
            if r is None:
                return None
            best = max(best, r)
        return best

    # Pre-build neighbor rings ONCE to avoid repeated 2nd-ring traversals
    _nb_ring1 = {}   # cell_id → set of 1st-ring neighbor IDs (excl co-sector)
    _nb_ring2 = {}   # cell_id → set of 2nd-ring neighbor IDs (excl co-sector, excl 1st)
    _prebuild_total = len(sorted_cells)
    _prebuild_step = max(1, _prebuild_total // 40)
    for _pre_idx, _pre_cid in enumerate(sorted_cells):
        if progress_callback and _pre_idx % _prebuild_step == 0:
            progress_callback(int(_pre_idx / _prebuild_total * 25),
                              f'Komşuluk haritası kuruluyor… {_pre_idx}/{_prebuild_total}')
        _my_sec = cell_to_sector.get(str(_pre_cid))
        _check_cells = _rsi_group(_pre_cid)
        _r1 = set()
        _r2 = set()
        for _src in _check_cells:
            for _nb in neighbors.get(_src, set()):
                _nb_id = str(_nb)
                if _my_sec is not None and cell_to_sector.get(_nb_id) == _my_sec:
                    continue
                _r1.add(_nb_id)
                for _nb2 in neighbors.get(_nb, set()):
                    _nb2_id = str(_nb2)
                    if _my_sec is not None and cell_to_sector.get(_nb2_id) == _my_sec:
                        continue
                    if _nb2_id != _pre_cid:
                        _r2.add(_nb2_id)
        _r2 -= _r1  # only true 2nd-ring (not already in 1st)
        _nb_ring1[_pre_cid] = _r1
        _nb_ring2[_pre_cid] = _r2

    def _occupied_roots(cell_id, include_2nd_ring=True):
        """Root indices used by assigned neighbours in the same root space.
        A neighbour with a different L_RA cannot clash (detect_rsi_collisions
        skips such pairs too), so it does not constrain this cell."""
        my_nzc = cell_info.get(cell_id, {}).get('nzc', NZC_LONG)
        rings = [_nb_ring1.get(cell_id, set())]
        if include_2nd_ring:
            rings.append(_nb_ring2.get(cell_id, set()))
        occ = set()
        for ring in rings:
            for nb_id in ring:
                nb_rsi = assigned.get(nb_id)
                if nb_rsi is None:
                    continue
                nb_info = cell_info.get(nb_id, {})
                if nb_info.get('nzc', NZC_LONG) != my_nzc:
                    continue
                nb_rn = assigned_rn.get(nb_id, nb_info.get('roots_needed', 1))
                nb_mx = nb_info.get('max_rsi', max_rsi)
                for i in range(nb_rn):
                    occ.add((nb_rsi + i) % nb_mx)
        return occ

    # --- reuse distance (rsi_strategy='max_reuse') ---------------------------
    _ids_all = df['cell_id'].astype(str).tolist()
    _lat = dict(zip(_ids_all, pd.to_numeric(df['latitude'], errors='coerce')))
    _lon = dict(zip(_ids_all, pd.to_numeric(df['longitude'], errors='coerce')))
    _space_of = {c: (carrier_map.get(c, CARRIER_UNKNOWN), cell_info[c]['nzc'])
                 for c in cell_info}
    _members_by_space = defaultdict(list)
    for c, sp in _space_of.items():
        _members_by_space[sp].append(c)

    def _reuse_profile(m, cell_mx):
        """Per root: distance (km) from m to the nearest assigned cell in m's
        carrier and root space that occupies that root (inf = unused)."""
        prof = np.full(cell_mx, np.inf)
        if pd.isna(_lat.get(m)) or pd.isna(_lon.get(m)):
            return prof
        sec = cell_to_sector.get(m)
        xs = [x for x in _members_by_space[_space_of[m]]
              if x != m and assigned.get(x) is not None
              and (sec is None or cell_to_sector.get(x) != sec)
              and pd.notna(_lat.get(x)) and pd.notna(_lon.get(x))]
        if not xs:
            return prof
        dist = _haversine_vec(_lat[m], _lon[m], [_lat[x] for x in xs], [_lon[x] for x in xs])
        starts = np.array([assigned[x] for x in xs], dtype=int)
        rns = np.array([assigned_rn.get(x) or cell_info[x]['roots_needed'] for x in xs], dtype=int)
        rep = np.repeat(np.arange(len(xs)), rns)
        offs = np.arange(int(rns.sum())) - np.repeat(np.cumsum(rns) - rns, rns)
        np.minimum.at(prof, (starts[rep] + offs) % cell_mx, dist[rep])
        return prof

    def _window_min(prof, rn):
        """w[c] = min(prof[c .. c+rn-1]) with wrap-around."""
        w = prof.copy()
        for i in range(1, rn):
            w = np.minimum(w, np.roll(prof, -i))
        return w

    def _find_free_rsi(occupied, group, cell_mx):
        """A start whose whole root range, for every cell of `group`, misses
        `occupied`: the lowest such start ('first_fit'), or the one whose
        nearest reuse is farthest away ('max_reuse').
        Returns (start, roots) or (None, None)."""
        restricted_group = any(cell_info[m]['restricted'] for m in group)
        if not restricted_group:
            rn = max(cell_info[m]['roots_needed'] for m in group)
            forbidden = set()
            for r in occupied:
                for k in range(rn):
                    forbidden.add((r - k) % cell_mx)
            clean = [c for c in range(cell_mx) if c not in forbidden]
            if not clean:
                return None, None
            if rsi_strategy == 'first_fit':
                return clean[0], rn
            w = None
            for m in group:
                wm = _window_min(_reuse_profile(m, cell_mx), cell_info[m]['roots_needed'])
                w = wm if w is None else np.minimum(w, wm)
            vals = w[np.array(clean)]
            return clean[int(np.argmax(vals))], rn   # ties -> lowest start
        # Restricted: the reservation depends on the start, walk the candidates.
        profs = ({m: _reuse_profile(m, cell_mx) for m in group}
                 if rsi_strategy != 'first_fit' else None)
        best = (None, None, -1.0)
        for c in range(cell_mx):
            rn = _group_rn_at(group, c)
            if rn is None or rn > cell_mx:
                continue
            if any(((c + i) % cell_mx) in occupied for i in range(rn)):
                continue
            if profs is None:
                return c, rn
            reuse = min(float(profs[m][(c + np.arange(_rn_at(m, c))) % cell_mx].min())
                        for m in group)
            if reuse > best[2]:
                best = (c, rn, reuse)
                if reuse == float('inf'):
                    break
        return best[0], best[1]

    def _assign(group, start):
        for m in group:
            assigned[m] = start
            assigned_rn[m] = _rn_at(m, start)

    already_assigned_by_sector = set()  # cells assigned via sector propagation
    _rsi_total = len(sorted_cells)
    _assign_step = max(1, _rsi_total // 50)  # ~2% increments → frequent bar updates

    for _rsi_i, cid in enumerate(sorted_cells):
        if progress_callback and _rsi_i % _assign_step == 0:
            progress_callback(25 + int(_rsi_i / _rsi_total * 60),
                              f'RSI atanıyor… {_rsi_i}/{_rsi_total}')
        # Skip if already assigned by sector propagation
        if cid in already_assigned_by_sector:
            continue

        group = [cid] + [m for m in _rsi_group(cid) if m != cid and m not in assigned]
        cell_max_rsi = cell_info[cid].get('max_rsi', max_rsi)

        found, _ = _find_free_rsi(_occupied_roots(cid), group, cell_max_rsi)
        if found is None:
            # Fallback: relax 2nd-ring constraint (still sector-aware)
            found, _ = _find_free_rsi(_occupied_roots(cid, include_2nd_ring=False),
                                      group, cell_max_rsi)
        if found is None:
            assigned[cid] = None
            continue
        _assign(group, found)
        already_assigned_by_sector.update(m for m in group if m != cid)

    # ------------------------------------------------------------------
    # RSI Repair Pass: detect remaining overlaps and fix them
    # Handles: (1) cells that got None, (2) sector-propagated conflicts
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(85, 'RSI onarım geçişi…')

    def _root_set(cid):
        start = assigned.get(cid)
        if start is None:
            return set()
        info = cell_info[cid]
        rn = assigned_rn.get(cid, info['roots_needed'])
        return {(start + i) % info.get('max_rsi', max_rsi) for i in range(rn)}

    def _has_rsi_overlap(cid_a, cid_b):
        """Do two assigned cells share a root, in the same root space?"""
        if assigned.get(cid_a) is None or assigned.get(cid_b) is None:
            return False
        if cell_info.get(cid_a, {}).get('nzc') != cell_info.get(cid_b, {}).get('nzc'):
            return False
        return bool(_root_set(cid_a) & _root_set(cid_b))

    for _repair_pass in range(5):
        # Detect remaining RSI overlaps among 1st-ring neighbors
        _rsi_violations = []
        _rsi_seen = set()
        for cid_a in sorted_cells:
            if assigned.get(cid_a) is None:
                continue
            sec_a = cell_to_sector.get(str(cid_a))
            for nb in neighbors.get(cid_a, set()):
                nb_id = str(nb)
                if assigned.get(nb_id) is None:
                    continue
                pair = tuple(sorted([cid_a, nb_id]))
                if pair in _rsi_seen:
                    continue
                _rsi_seen.add(pair)
                # Skip co-sector
                sec_b = cell_to_sector.get(nb_id)
                if sec_a is not None and sec_a == sec_b:
                    continue
                if _has_rsi_overlap(cid_a, nb_id):
                    _rsi_violations.append(pair)

        if not _rsi_violations:
            break  # all clean

        # Fix each violation: reassign the cell with fewer neighbors
        for va, vb in _rsi_violations:
            if not _has_rsi_overlap(va, vb):
                continue  # already fixed by earlier repair

            # Pick the cell with fewer neighbors (easier to reassign)
            deg_a = cell_info.get(va, {}).get('degree', 0)
            deg_b = cell_info.get(vb, {}).get('degree', 0)
            to_fix = vb if deg_b <= deg_a else va

            # The whole same-L_RA sector group moves together
            fix_group = _rsi_group(to_fix)
            rep = fix_group[0]
            cell_mx = cell_info[rep].get('max_rsi', max_rsi)

            # Temporarily remove current assignment to avoid self-blocking
            old = {fc: (assigned.get(fc), assigned_rn.get(fc)) for fc in fix_group}
            for fc in fix_group:
                assigned[fc] = None

            occ = _occupied_roots(rep, include_2nd_ring=False)
            found, _ = _find_free_rsi(occ, fix_group, cell_mx)

            if found is not None:
                _assign(fix_group, found)
            else:
                # Restore old assignment (couldn't improve)
                for fc in fix_group:
                    assigned[fc] = old[fc][0]
                    if old[fc][1] is not None:
                        assigned_rn[fc] = old[fc][1]

    # Also try to assign RSI for cells that got None
    if progress_callback:
        progress_callback(95, 'Kalan hücreler atanıyor…')
    for cid in sorted_cells:
        if assigned.get(cid) is not None:
            continue
        group = [cid] + [m for m in _rsi_group(cid) if m != cid and assigned.get(m) is None]
        cell_mx = cell_info[cid].get('max_rsi', max_rsi)
        occ = _occupied_roots(cid, include_2nd_ring=False)
        found, _ = _find_free_rsi(occ, group, cell_mx)
        if found is not None:
            _assign(group, found)

    # Nearest reuse of each cell's roots in the final plan (reviewability, Ö-2)
    _reuse = rsi_reuse_distances(
        df, {c: s for c, s in assigned.items() if s is not None}, technology,
        carrier_map, cell_to_sector)

    # Build result table
    rows = []
    for cid in sorted_cells:
        info = cell_info[cid]
        planned = assigned.get(cid)
        rn = (assigned_rn.get(cid, info['roots_needed']) if planned is not None
              else info['roots_needed'])
        cur = info['current_rsi']
        cell_mx = info.get('max_rsi', max_rsi)
        changed = (planned != cur) if (planned is not None and cur is not None) else True
        _rk, _rw = _reuse.get(cid, (float('inf'), None))
        rows.append({
            'cell_id': cid,
            'current_rsi': cur if cur is not None else '—',
            'planned_rsi': planned if planned is not None else '—',
            'ncs': info['ncs'],
            'roots_needed': rn,
            'planned_range': f"{planned}-{(planned+rn-1)%cell_mx}" if planned is not None else '—',
            'cell_range_km': info['cell_range_km'],
            'reuse_km': round(_rk, 2) if _rk != float('inf') else None,
            'reuse_with': _rw or '—',
            'changed': '✅ Değişti' if changed else '— Aynı',
            'reason': (f"RSI {cur}→{planned}: {rn} root, aralık {planned}-{(planned+rn-1)%cell_mx}"
                       if planned is not None and changed
                       else ('Mevcut RSI uygun' if not changed
                             else 'RSI atanamadı (alan dolu)'))
        })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        for col in ['current_rsi','planned_rsi']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
    result_df = enrich_df_with_sector_info(result_df)
    result_df.attrs['rsi_strategy_used'] = rsi_strategy

    # Spreading roots over the whole space can fragment it: a large cell
    # (20-30 roots) then finds no contiguous free window.  On the real Burdur
    # network with its handover list, max_reuse left 10 cells unassigned
    # where first_fit assigned all.  So: retry with the stuck cells first,
    # and never return a plan that leaves more cells without an RSI than
    # first_fit would.  Ties go to max_reuse (better reuse distance).
    unassigned = [c for c in sorted_cells if assigned.get(c) is None]
    if rsi_strategy == 'max_reuse' and unassigned and _priority_cells is None:
        _n = lambda plan: int((plan['planned_rsi'] == '—').sum())
        options = [(len(unassigned), 0, result_df, None)]
        retry = plan_rsi_network(*_call_args, rsi_strategy='max_reuse',
                                 _priority_cells=set(unassigned))
        options.append((_n(retry), 1, retry,
                        f"{len(unassigned)} hücre ilk geçişte atanamadı; öncelik verilerek "
                        f"yeniden planlandı ({_n(retry)} atanamayan)."))
        if _n(retry):
            ff = plan_rsi_network(*_call_args, rsi_strategy='first_fit')
            ff.attrs['rsi_strategy_used'] = 'first_fit'
            options.append((_n(ff), 2, ff,
                            f"En uzak yeniden kullanım en iyi {min(len(unassigned), _n(retry))} "
                            f"hücreyi atayamadı, ilk uygun {_n(ff)} — ilk uygun planı kullanıldı."))
        best = min(options, key=lambda o: (o[0], o[1]))
        if best[3]:
            best[2].attrs['fallback_reason'] = best[3]
        return best[2]
    return result_df


def plan_pci_network(df, neighbors, technology='LTE',
                     check_mod3=True, check_mod6=True, check_mod30=None,
                     sector_groups=None, cell_to_sector=None,
                     nbr_attempts=None, progress_callback=None,
                     check_mod4=False,
                     sa_iterations_override=0,
                     reserved_pci_start=None, reserved_pci_end=None,
                     carrier_map=None, planning_scope='sector',
                     attempt_weighting=True):
    """Assign PCI values using **Constrained Simulated Annealing** optimisation.

    Algorithm (based on professional PCI planning tool research):
      Phase 0 — Mod3 class pre-assignment:
        Each physical site's sectors are assigned distinct mod3 classes
        (0, 1, 2) via graph coloring, making co-site PSS collision
        structurally impossible during optimisation.
      1. Graph coloring formulation — cells are vertices, neighbor relations
         are edges, PCI values are colors (constrained to assigned mod3 class).
      2. Weighted energy function balances collision, confusion, mod3/4/6/30
         penalties.
      3. Simulated Annealing explores the solution space probabilistically,
         accepting worse moves early (high T) to escape local minima, then
         converging to a near-optimal solution as temperature cools.
      4. Best-ever solution is tracked and returned.

    Technology-aware:
      LTE → check_mod6, check_mod30 respected; mod4 ignored
      NR  → check_mod4 auto-enabled; mod6/mod30 auto-disabled

    Returns DataFrame with:
        cell_id, current_pci, planned_pci, pss, sss, changed, reason
    """
    technology = norm_tech(technology)  # UI = tek otorite
    if sector_groups is None:
        sector_groups = {}
    if cell_to_sector is None:
        cell_to_sector = {}
    if nbr_attempts is None:
        nbr_attempts = {}
    if carrier_map is None:
        carrier_map = build_carrier_map(df)
    if planning_scope not in PLANNING_SCOPES:
        raise ValueError(f"planning_scope '{planning_scope}' gecersiz; "
                         f"secenekler: {PLANNING_SCOPES}")
    # 'carrier' scope: every carrier of a sector gets its own PCI decision.
    # 'sector' scope (default): one PCI for the whole physical sector across
    # all of its carriers — the operator's current strategy.
    if planning_scope == 'carrier':
        sector_groups, cell_to_sector = split_sector_groups_by_carrier(
            sector_groups, cell_to_sector, carrier_map)

    # Technology-aware: NR uses mod4, LTE uses mod6/mod30
    # NR'de mod30 ZORLA kapatilmaz: 30-gruplu low-PAPR dizi yapisi NR'de de
    # vardir, ama PCI'ya bagli olmasi kosulludur (TS 38.211 §6.3.2.2.1 —
    # hoppingId konfigure edilmisse dizi grubu PCI'dan kopar; §6.4.1.4.2 —
    # SRS zaten hep kendi sequenceId'sini kullanir).  Bu yuzden varsayilan
    # kapali ama cagiran acikca acabilir.
    if check_mod30 is None:
        check_mod30 = (technology != 'NR')
    if technology == 'NR':
        check_mod4 = True
        check_mod6 = False      # NR'de CRS yok

    max_pci = pci_count(technology)
    pci_map = dict(zip(df['cell_id'].astype(str), df['pci']))

    # Build co-site set
    co_site_set = build_co_site_set(df, cell_to_sector)

    cell_ids = [str(c) for c in df['cell_id']]
    cell_set = set(cell_ids)
    n_cells = len(cell_ids)

    # ------------------------------------------------------------------
    # Build efficient data structures for SA
    # ------------------------------------------------------------------
    # Sector leaders: for each sector group, pick one leader; others mirror it
    sector_leader = {}   # sector_key → leader cell_id
    cell_is_follower = set()  # cells that just copy leader's PCI
    for sec_key, members in sector_groups.items():
        leader = None
        for m in members:
            if m in cell_set:
                leader = m
                break
        if leader:
            sector_leader[sec_key] = leader
            for m in members:
                if m != leader and m in cell_set:
                    cell_is_follower.add(m)

    # Independent cells = those we actually optimise (leaders + ungrouped)
    independent_cells = [c for c in cell_ids if c not in cell_is_follower]
    n_indep = len(independent_cells)
    indep_idx = {c: i for i, c in enumerate(independent_cells)}

    # For each independent cell, list of all cells sharing its PCI (itself + co-sector followers)
    def _sector_members(cid):
        """Return list of all cells that share PCI with cid (including cid)."""
        sec_key = cell_to_sector.get(cid)
        if not sec_key:
            return [cid]
        return [m for m in sector_groups.get(sec_key, [cid]) if m in cell_set]

    sector_members_cache = {}
    for cid in independent_cells:
        sector_members_cache[cid] = _sector_members(cid)

    # Per-pair traffic weight.  _pair_weight keeps RAW attempt counts, used by
    # the Phase 0 site colouring where an absolute comparison is wanted.
    # _traffic_w is the compressed weight used inside the SA objective.
    _has_attempts = bool(nbr_attempts)
    def _pair_weight(a, b):
        if not _has_attempts:
            return 1
        pair = tuple(sorted([str(a), str(b)]))
        return nbr_attempts.get(pair, 1)

    if attempt_weighting and _has_attempts:
        _traffic_w, _traffic_stats = build_attempt_weights(nbr_attempts)
    else:
        _traffic_w, _traffic_stats = (lambda a, b: 1.0), {'enabled': False}

    # Pre-compute co-site membership as a fast set for O(1) lookup
    # Use ALL same-site detection methods (matching _count_cosite) so the
    # energy function penalises the same pairs that the scoring counts.
    _co_site_flat = set()
    for pair in co_site_set:
        items = tuple(pair)
        _co_site_flat.add((items[0], items[1]))
        _co_site_flat.add((items[1], items[0]))

    # Expand with site_id column matches
    if 'site_id' in df.columns:
        _sid_groups = defaultdict(list)
        for _, r in df.iterrows():
            sid = str(r.get('site_id', '')).strip()
            cid_s = str(r['cell_id'])
            if sid and sid != 'nan' and sid != '' and cid_s in cell_set:
                _sid_groups[sid].append(cid_s)
        for _sg in _sid_groups.values():
            for _i in range(len(_sg)):
                for _j in range(_i + 1, len(_sg)):
                    a, b = _sg[_i], _sg[_j]
                    # Exclude co-sector pairs (naming convention primary)
                    if _is_co_sector_by_id(a, b):
                        continue
                    sa, sb = cell_to_sector.get(a), cell_to_sector.get(b)
                    if sa and sa == sb:
                        continue
                    _co_site_flat.add((a, b))
                    _co_site_flat.add((b, a))

    # Expand with cell ID naming convention matches
    _cid_name_groups = defaultdict(list)
    for cid in cell_ids:
        sn = _extract_site_name(cid)
        if sn is not None:
            _cid_name_groups[sn].append(cid)
    for _ng in _cid_name_groups.values():
        for _i in range(len(_ng)):
            for _j in range(_i + 1, len(_ng)):
                a, b = _ng[_i], _ng[_j]
                # Exclude co-sector pairs (naming convention primary)
                if _is_co_sector_by_id(a, b):
                    continue
                sa, sb = cell_to_sector.get(a), cell_to_sector.get(b)
                if sa and sa == sb:
                    continue
                _co_site_flat.add((a, b))
                _co_site_flat.add((b, a))

    # Build neighbor lists excluding co-sector (as sets of cell_id strings)
    # Carrier scoping (K-1).  Two adjacency views are needed:
    #   nb_excl_cosector - SAME-carrier neighbours.  Collision, mod-N and
    #     co-site PSS checks only make sense within one carrier.
    #   nb_conf - ALL-carrier neighbours, used only for the confusion
    #     traversal: a cell legitimately reports inter-frequency neighbours,
    #     and two of those sharing a PCI is a real ambiguity on THEIR carrier.
    #     _cell_energy still requires the ambiguous pair to share a carrier.
    nb_excl_cosector = {}
    nb_conf = {}
    for cid in cell_ids:
        my_sec = cell_to_sector.get(cid)
        nbs = set()
        nbs_all = set()
        for nb in neighbors.get(cid, set()):
            nb_id = str(nb)
            # Exclude co-sector: naming convention (primary) + cell_to_sector (fallback)
            if _is_co_sector_by_id(cid, nb_id):
                continue
            if my_sec and cell_to_sector.get(nb_id) == my_sec:
                continue
            nbs_all.add(nb_id)
            if same_carrier(carrier_map, cid, nb_id):
                nbs.add(nb_id)
        nbs.discard(cid)  # never be neighbor of yourself
        nbs_all.discard(cid)
        nb_excl_cosector[cid] = nbs
        nb_conf[cid] = nbs_all

    # Reverse-PCI index for efficient confusion detection.
    # Instead of pre-building O(N*D^2) 2-hop sets, maintain a live
    # mapping: pci_value -> set of cell_ids assigned that PCI.
    # Confusion for cell A (PCI=p) via its neighbor B:
    #   "does B have another neighbor with PCI=p?"
    #   = (neighbors_of_B & pci_to_cells[p]) - {A} - co_sector_of_A
    # This is O(min(|neighbors_of_B|, |same_pci_cells|)) per neighbor.
    pci_to_cells = defaultdict(set)  # pci_value -> {cell_id, ...}

    # ------------------------------------------------------------------
    # Energy function: count weighted violations for current assignment
    # We compute DELTA energy when changing one independent cell's PCI.
    # ------------------------------------------------------------------
    # Penalty weights (aligned with health score priorities)
    W_COLLISION   = 100.0   # Most critical — must be eliminated
    W_CO_SITE_M3  = 200.0   # Co-site PSS collision — outdoor↔outdoor: hard constraint
    W_CO_SITE_M3_INDOOR = 80.0   # Co-site PSS collision involving indoor — softer but significant
    W_CONFUSION   = 30.0    # Handover ambiguity — high weight to prevent increase
    W_MOD3        = 8.0 if check_mod3 else 0.0
    W_MOD4        = 2.5 if check_mod4 else 0.0   # NR SSB DMRS interference
    W_MOD6        = 1.5 if check_mod6 else 0.0
    W_MOD30       = 0.5 if check_mod30 else 0.0

    # Build indoor cell set for indoor-aware co-site mod3 penalty
    _indoor_cells = set()
    for cid in cell_ids:
        if _is_indoor_cell(cid):
            _indoor_cells.add(cid)

    # Build set of co-site pairs where at least one cell is indoor
    # (these get lower mod3 penalty)
    _co_site_has_indoor = set()  # set of (a, b) tuples
    for pair in _co_site_flat:
        a, b = pair
        if a in _indoor_cells or b in _indoor_cells:
            _co_site_has_indoor.add(pair)

    # Current assignment: start from existing PCIs.
    # A PCI outside the technology's range (e.g. an NR PCI in an LTE run) is
    # NOT usable as a starting point — treat it as unassigned so the planner
    # replaces it instead of carrying it through to the output.
    assignment = {}  # cell_id → pci (int)
    out_of_range_seed = []
    for cid in cell_ids:
        cur = pci_map.get(cid)
        if cur is not None and not pd.isna(cur) and pci_in_range(cur, technology):
            assignment[cid] = int(cur)
        else:
            if cur is not None and not pd.isna(cur):
                out_of_range_seed.append(str(cid))
            # Unassigned / out-of-range cells: give a random valid initial PCI
            assignment[cid] = random.randint(0, max_pci - 1)

    # Ensure co-sector consistency: followers copy leader
    for cid in independent_cells:
        pci_val = assignment[cid]
        for member in sector_members_cache[cid]:
            assignment[member] = pci_val

    # Initialize reverse-PCI index from current assignment
    for cid in cell_ids:
        p = assignment.get(cid)
        if p is not None:
            pci_to_cells[p].add(cid)

    # Pre-compute candidate PCI lists per mod3 class
    pci_by_mod3 = {0: [], 1: [], 2: []}
    _reserved = set()
    if reserved_pci_start is not None and reserved_pci_end is not None:
        _reserved = set(range(int(reserved_pci_start), int(reserved_pci_end) + 1))
    for p in range(max_pci):
        if p in _reserved:
            continue
        pci_by_mod3[p % 3].append(p)

    # ------------------------------------------------------------------
    # Phase 0: Constrained mod3 class pre-assignment
    # Assign mod3 class (0, 1, 2) to each site's sectors so that co-site
    # cells always have different PSS (PCI mod 3).  During SA, each cell
    # may only receive PCIs from its assigned class — this makes co-site
    # mod3 violations structurally impossible.
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(1, 'Faz 0: Site sektörlerine mod3 sınıfları atanıyor…')

    cell_mod3_class = {}  # cell_id → int (0, 1, or 2); absent = unconstrained

    # Build site clusters via Union-Find — use ALL same-site detection
    # methods (matching _count_cosite in run_full_analysis) to ensure
    # no co-site pair is missed.
    _sp = {cid: cid for cid in cell_ids}

    def _sfind(x):
        while _sp[x] != x:
            _sp[x] = _sp[_sp[x]]
            x = _sp[x]
        return x

    def _sunion(a, b):
        ra, rb = _sfind(a), _sfind(b)
        if ra != rb:
            _sp[ra] = rb

    # Method 1: Co-site pairs from location proximity (≤50m)
    for pair in co_site_set:
        items = tuple(pair)
        _sunion(items[0], items[1])

    # Method 2: Co-sector members → same physical site
    for _sk, _members in sector_groups.items():
        _valid = [m for m in _members if m in cell_set]
        for _i in range(1, len(_valid)):
            _sunion(_valid[0], _valid[_i])

    # Method 3: site_id column — cells with same non-empty site_id are same site
    _sid_map = {}
    if 'site_id' in df.columns:
        for _, r in df.iterrows():
            sid = str(r.get('site_id', '')).strip()
            cid = str(r['cell_id'])
            if sid and sid != 'nan' and sid != '' and cid in cell_set:
                _sid_map.setdefault(sid, []).append(cid)
        for _sid_cells in _sid_map.values():
            for _i in range(1, len(_sid_cells)):
                _sunion(_sid_cells[0], _sid_cells[_i])

    # Method 4: Cell ID naming convention (_is_same_site_by_id)
    _name_groups = defaultdict(list)
    for cid in cell_ids:
        sn = _extract_site_name(cid)
        if sn is not None:
            _name_groups[sn].append(cid)
    for _ng_cells in _name_groups.values():
        for _i in range(1, len(_ng_cells)):
            _sunion(_ng_cells[0], _ng_cells[_i])

    # Collect site clusters
    _site_clusters = defaultdict(set)
    for cid in cell_ids:
        _site_clusters[_sfind(cid)].add(cid)

    from itertools import permutations as _perms

    for _site_root, _site_cells in _site_clusters.items():
        if len(_site_cells) <= 1:
            continue  # Single-cell site: no co-site constraint

        # Identify sector units: co-sector groups + singletons
        _sec_units = {}
        _ungrouped = []
        for cid in _site_cells:
            sk = cell_to_sector.get(cid)
            if sk:
                _sec_units.setdefault(sk, []).append(cid)
            else:
                _ungrouped.append(cid)
        _unit_list = list(_sec_units.values())
        for cid in _ungrouped:
            _unit_list.append([cid])

        n_units = len(_unit_list)
        if n_units <= 1:
            continue  # All cells same sector → share PCI anyway

        # Classify each unit as indoor or outdoor
        # A unit is indoor if ALL its cells are indoor
        _unit_is_indoor = []
        for unit_cells in _unit_list:
            all_indoor = all(cid in _indoor_cells for cid in unit_cells)
            _unit_is_indoor.append(all_indoor)

        # Separate outdoor and indoor units
        _outdoor_units = [(i, _unit_list[i]) for i in range(n_units) if not _unit_is_indoor[i]]
        _indoor_units = [(i, _unit_list[i]) for i in range(n_units) if _unit_is_indoor[i]]

        n_outdoor = len(_outdoor_units)

        if n_outdoor <= 3 and n_outdoor >= 1:
            # ── OUTDOOR sectors get distinct mod3 classes (0,1,2) ──
            # Try all permutations to find best mod3 assignment for outdoor
            # units, minimizing inter-site neighbor conflicts.
            # Indoor units are left UNCONSTRAINED (SA will optimize them).
            best_perm = tuple(range(n_outdoor))
            best_conflict = float('inf')
            for perm in _perms([0, 1, 2]):
                perm_slice = perm[:n_outdoor]
                conflict = 0
                for oi, (ui, unit_cells) in enumerate(_outdoor_units):
                    m3c = perm_slice[oi]
                    for cid in unit_cells:
                        for nb in nb_excl_cosector.get(cid, set()):
                            if nb in _site_cells:
                                continue  # intra-site, skip
                            nb_m3c = cell_mod3_class.get(nb)
                            if nb_m3c is None:
                                nb_pci = assignment.get(nb)
                                nb_m3c = nb_pci % 3 if nb_pci is not None else -1
                            if nb_m3c == m3c:
                                conflict += _pair_weight(cid, nb)
                if conflict < best_conflict:
                    best_conflict = conflict
                    best_perm = perm_slice
            # Apply outdoor mod3 classes
            for oi, (ui, unit_cells) in enumerate(_outdoor_units):
                m3c = best_perm[oi]
                for cid in unit_cells:
                    cell_mod3_class[cid] = m3c
            # Indoor units: assign remaining mod3 classes to avoid co-site PSS conflict
            # With ≤3 outdoor, there are leftover classes for indoor units
            _used_classes = set(best_perm)
            _avail_classes = [c for c in [0, 1, 2] if c not in _used_classes]
            _all_classes = _avail_classes + list(best_perm)  # prefer unused first
            for ii, (ui, unit_cells) in enumerate(_indoor_units):
                # Cycle through available classes; if >3 total units, some will share
                m3c = _all_classes[ii % len(_all_classes)] if _all_classes else ii % 3
                for cid in unit_cells:
                    cell_mod3_class[cid] = m3c
        elif n_outdoor == 0:
            # Pure indoor site — constrain up to 3 units as normal
            _indoor_only_list = [uc for _, uc in _indoor_units]
            if len(_indoor_only_list) <= 3:
                best_perm = (0, 1, 2)
                best_conflict = float('inf')
                for perm in _perms([0, 1, 2]):
                    conflict = 0
                    for ui_idx, unit_cells in enumerate(_indoor_only_list):
                        m3c = perm[ui_idx]
                        for cid in unit_cells:
                            for nb in nb_excl_cosector.get(cid, set()):
                                if nb in _site_cells:
                                    continue
                                nb_m3c = cell_mod3_class.get(nb)
                                if nb_m3c is None:
                                    nb_pci = assignment.get(nb)
                                    nb_m3c = nb_pci % 3 if nb_pci is not None else -1
                                if nb_m3c == m3c:
                                    conflict += _pair_weight(cid, nb)
                    if conflict < best_conflict:
                        best_conflict = conflict
                        best_perm = perm
                for ui_idx, unit_cells in enumerate(_indoor_only_list):
                    m3c = best_perm[ui_idx]
                    for cid in unit_cells:
                        cell_mod3_class[cid] = m3c
            else:
                # >3 indoor units: cycle through mod3 classes
                for ui_idx, unit_cells in enumerate(_indoor_only_list):
                    m3c = ui_idx % 3
                    for cid in unit_cells:
                        cell_mod3_class[cid] = m3c
        else:
            # >3 outdoor units — unusual; cycle through mod3 classes
            all_units = [uc for _, uc in (_outdoor_units + _indoor_units)]
            for ui_idx, unit_cells in enumerate(all_units):
                m3c = ui_idx % 3
                for cid in unit_cells:
                    cell_mod3_class[cid] = m3c

    # ------------------------------------------------------------------
    # Build per-leader co-site leader map for hard collision prevention.
    # For each independent cell (leader), list all OTHER leaders on the
    # same physical site.  During SA, PCIs used by co-site leaders are
    # forbidden — this makes co-site collision structurally impossible.
    # ------------------------------------------------------------------
    _co_site_leaders = defaultdict(set)
    for _site_root, _site_cells in _site_clusters.items():
        site_leaders = [c for c in _site_cells if c not in cell_is_follower]
        if len(site_leaders) <= 1:
            continue
        for _sl in site_leaders:
            for _sl2 in site_leaders:
                if _sl != _sl2:
                    _co_site_leaders[_sl].add(_sl2)

    def _co_site_forbidden(cid):
        """Return set of PCIs currently used by co-site leaders (hard forbidden)."""
        fb = set()
        for co_l in _co_site_leaders.get(cid, set()):
            p = assignment.get(co_l)
            if p is not None:
                fb.add(p)
        return fb

    # Snapshot the ORIGINAL assignment before Phase 0 rewrites it.  Phase 0
    # gives a random PCI to every sector whose current PCI does not match its
    # newly assigned mod3 class, which on a large network is most of them.
    # Without this snapshot the input plan is never a candidate, so a run that
    # fails to converge can return something worse than what it was given.
    original_assignment = dict(assignment)
    original_usable = all(pci_in_range(v, technology) and v not in _reserved
                          for v in original_assignment.values())

    # Fix initial PCIs to respect assigned mod3 classes,
    # clear reserved range, and prevent co-site collisions.
    for cid in independent_cells:
        expected_m3 = cell_mod3_class.get(cid)
        cur_p = assignment[cid]
        needs_fix = False
        if not pci_in_range(cur_p, technology):
            needs_fix = True
        if expected_m3 is not None and cur_p % 3 != expected_m3:
            needs_fix = True
        if cur_p in _reserved:
            needs_fix = True
        co_site_fb = _co_site_forbidden(cid)
        if cur_p in co_site_fb:
            needs_fix = True
        if not needs_fix:
            continue
        old_p = cur_p
        members = sector_members_cache[cid]
        # Collect PCIs used by neighbors (to avoid collisions)
        nb_pcis = set()
        for m in members:
            for nb_id in nb_excl_cosector.get(m, set()):
                nb_p = assignment.get(nb_id)
                if nb_p is not None:
                    nb_pcis.add(nb_p)
        nb_pcis |= co_site_fb
        # Pick from assigned class, collision-free if possible
        if expected_m3 is not None:
            pool = pci_by_mod3[expected_m3]
        else:
            pool = [p for p in range(max_pci) if p not in _reserved]
        safe = [p for p in pool if p not in nb_pcis]
        new_p = random.choice(safe) if safe else random.choice(pool)
        for m in members:
            pci_to_cells[old_p].discard(m)
            assignment[m] = new_p
            pci_to_cells[new_p].add(m)

    _EMPTY = frozenset()

    def _cell_energy(cid, pci_val, asgn):
        """Compute energy contribution of cell `cid` having `pci_val`.
        Only counts violations with ASSIGNED neighbors (1-hop and 2-hop).
        Returns the energy as a float."""
        e = 0.0
        m3 = pci_val % 3
        m4 = pci_val % 4
        m6 = pci_val % 6
        m30 = pci_val % 30
        # 1-hop checks: collision, co-site mod3, mod3/4/6/30
        # Every pairwise penalty is scaled by the traffic on that relation, so
        # when a conflict is unavoidable the annealer puts it on a quiet pair
        # rather than on one carrying tens of thousands of handovers.
        # Co-site mod3 is deliberately NOT scaled: two sectors on one site must
        # have different PSS whatever their traffic.
        for nb_id in nb_excl_cosector.get(cid, set()):
            nb_pci = asgn.get(nb_id)
            if nb_pci is None:
                continue
            w = _traffic_w(cid, nb_id)
            if pci_val == nb_pci:
                e += W_COLLISION * w
            else:
                nb_m3 = nb_pci % 3
                if (cid, nb_id) in _co_site_flat and m3 == nb_m3:
                    if (cid, nb_id) in _co_site_has_indoor:
                        e += W_CO_SITE_M3_INDOOR  # indoor involved → soft
                    else:
                        e += W_CO_SITE_M3  # outdoor↔outdoor → hard
                if W_MOD3 > 0 and m3 == nb_m3:
                    e += W_MOD3 * w
                if W_MOD4 > 0 and m4 == nb_pci % 4:
                    e += W_MOD4 * w
                if W_MOD6 > 0 and m6 == nb_pci % 6:
                    e += W_MOD6 * w
                if W_MOD30 > 0 and m30 == nb_pci % 30:
                    e += W_MOD30 * w
        # 2-hop checks: confusion via reverse-PCI index.
        # For each neighbor B of cid, check if any cell sharing cid's PCI
        # is also a neighbor of B (=> 2-hop confusion pair).
        if W_CONFUSION > 0:
            same_pci = pci_to_cells.get(pci_val)
            if same_pci and len(same_pci) > 1:
                my_sec = cell_to_sector.get(cid)
                my_nbs = nb_conf.get(cid, _EMPTY)
                for nb_id in my_nbs:
                    nb_nbs = nb_conf.get(nb_id, _EMPTY)
                    # Use smaller set for intersection check
                    if len(same_pci) < len(nb_nbs):
                        for c2 in same_pci:
                            if c2 != cid and c2 in nb_nbs:
                                if not same_carrier(carrier_map, cid, c2):
                                    continue  # ambiguity lives within one carrier
                                if not (my_sec and cell_to_sector.get(c2) == my_sec):
                                    # the ambiguity is felt on whichever of the
                                    # two relations carries more traffic
                                    e += W_CONFUSION * max(_traffic_w(nb_id, cid),
                                                           _traffic_w(nb_id, c2))
                    else:
                        for c2 in nb_nbs:
                            if c2 != cid and c2 in same_pci:
                                if not same_carrier(carrier_map, cid, c2):
                                    continue
                                if not (my_sec and cell_to_sector.get(c2) == my_sec):
                                    e += W_CONFUSION * max(_traffic_w(nb_id, cid),
                                                           _traffic_w(nb_id, c2))
        return e

    def _group_energy(leader, pci_val, asgn):
        """Compute energy for a leader and all its co-sector members."""
        e = 0.0
        for member in sector_members_cache[leader]:
            e += _cell_energy(member, pci_val, asgn)
        return e

    # Compute initial total energy
    total_energy = 0.0
    for cid in independent_cells:
        total_energy += _group_energy(cid, assignment[cid], assignment)
    # Each undirected pair is counted from both sides, so divide by 2
    total_energy /= 2.0

    # ------------------------------------------------------------------
    # Simulated Annealing
    # ------------------------------------------------------------------
    # SA parameters — tuned for PCI planning problem
    # Scale iterations with network size but cap for performance
    if sa_iterations_override and sa_iterations_override > 0:
        sa_iterations = int(sa_iterations_override)
    else:
        # Adaptive: small clusters need far fewer iterations than the old
        # 300K floor; n*400 keeps behaviour identical for ≥750 cells.
        sa_iterations = min(max(30_000, n_indep * 400), 800_000)
    T_start = 50.0
    T_end = 0.01
    # Geometric cooling rate: T_end = T_start * alpha^iterations
    alpha = (T_end / T_start) ** (1.0 / sa_iterations)

    best_assignment = dict(assignment)
    best_energy = total_energy

    # Do no harm: if the plan we were handed is better than the post-Phase-0
    # state, it becomes the incumbent.  The planner may then only improve on it.
    original_energy = None
    if original_usable:
        _saved = dict(assignment)
        for k, v in original_assignment.items():
            assignment[k] = v
        original_energy = 0.0
        for cid in independent_cells:
            original_energy += _group_energy(cid, assignment[cid], assignment)
        original_energy /= 2.0
        for k, v in _saved.items():
            assignment[k] = v
        if original_energy < best_energy:
            best_assignment = dict(original_assignment)
            best_energy = original_energy

    T = T_start
    accept_count = 0
    improve_count = 0

    t_start = time.time()
    _sa_report_interval = max(1, sa_iterations // 20)  # report every 5%

    for iteration in range(sa_iterations):
        if progress_callback and iteration % _sa_report_interval == 0:
            progress_callback(int(iteration / sa_iterations * 95),
                              f'SA iterasyon {iteration:,}/{sa_iterations:,} — enerji: {best_energy:.0f}')
        # Pick a random independent cell
        idx = random.randint(0, n_indep - 1)
        cid = independent_cells[idx]
        old_pci = assignment[cid]

        # Generate a candidate PCI (different from current)
        # Hard constraint: exclude PCIs used by co-site leaders (prevents
        # co-site collision entirely).  Rejection loop — probability of
        # hitting a forbidden PCI is ~k/168 where k = co-site sectors (~2-4),
        # so typically resolves in 1-2 iterations.
        _co_fb = _co_site_forbidden(cid)
        _assigned_m3 = cell_mod3_class.get(cid)
        new_pci = old_pci  # will be overwritten
        for _pick_try in range(20):
            if _assigned_m3 is not None:
                _cand = random.choice(pci_by_mod3[_assigned_m3])
            else:
                if random.random() < 0.5:
                    cur_mod3 = old_pci % 3
                    other_classes = [m for m in [0, 1, 2] if m != cur_mod3]
                    chosen_class = random.choice(other_classes)
                    _cand = random.choice(pci_by_mod3[chosen_class])
                else:
                    _cand = random.randint(0, max_pci - 1)
            if _cand != old_pci and _cand not in _co_fb:
                new_pci = _cand
                break

        if new_pci == old_pci:
            T *= alpha
            continue

        # --- Delta energy calculation (efficient: only affected cells) ---
        # Old energy of this group
        old_group_e = _group_energy(cid, old_pci, assignment)

        # Temporarily apply new PCI + update reverse index
        members = sector_members_cache[cid]
        for m in members:
            pci_to_cells[old_pci].discard(m)
            assignment[m] = new_pci
            pci_to_cells[new_pci].add(m)

        # New energy of this group
        new_group_e = _group_energy(cid, new_pci, assignment)

        # Delta (accounting for double-counting: each pair seen from both sides)
        delta_e = (new_group_e - old_group_e) / 2.0

        # Metropolis acceptance criterion
        accept = False
        if delta_e <= 0:
            accept = True
        else:
            # Accept with probability exp(-delta_e / T)
            if T > 1e-10:
                prob = _math.exp(-delta_e / T)
                if random.random() < prob:
                    accept = True

        if accept:
            total_energy += delta_e
            accept_count += 1
            if total_energy < best_energy:
                best_energy = total_energy
                best_assignment = dict(assignment)
                improve_count += 1
                # Perfect solution (zero conflicts) — no better state exists
                if best_energy <= 1e-9:
                    break
        else:
            # Revert + restore reverse index
            for m in members:
                pci_to_cells[new_pci].discard(m)
                assignment[m] = old_pci
                pci_to_cells[old_pci].add(m)

        T *= alpha

    # Restore best assignment found
    assignment = best_assignment

    # Rebuild reverse-PCI index before the rotation pass
    pci_to_cells.clear()
    for cid in cell_ids:
        p = assignment.get(cid)
        if p is not None:
            pci_to_cells[p].add(cid)

    # ------------------------------------------------------------------
    # Site mod3-rotation pass
    # ------------------------------------------------------------------
    # Phase 0 colours each site's sectors with distinct mod3 classes and then
    # freezes them: during SA a cell may only take PCIs from its assigned class,
    # so SA has no lever on mod3 at all.  Rotating a whole site's classes
    # (0->1->2->0 and the other permutations) keeps every sector distinct — the
    # co-site PSS guarantee is untouched — but changes mod3 against the rest of
    # the network.  Measured on the real Samsun network at 3 km this takes mod3
    # from 4,483 to 4,227 (-5.7%).
    #
    # Each rotation is applied as a sequence of ordinary single-cell moves so it
    # reuses the same exact delta bookkeeping as SA, and the whole rotation is
    # reverted unless the TOTAL energy strictly improves.
    _ROTATIONS = ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))

    def _apply_pci(leader, new_pci):
        """Move a leader group to new_pci, returning the exact energy delta."""
        members = sector_members_cache[leader]
        old_pci = assignment[leader]
        if new_pci == old_pci:
            return 0.0
        e_before = _group_energy(leader, old_pci, assignment)
        for m in members:
            pci_to_cells[old_pci].discard(m)
            assignment[m] = new_pci
            pci_to_cells[new_pci].add(m)
        e_after = _group_energy(leader, new_pci, assignment)
        return e_after - e_before

    _site_leader_groups = defaultdict(list)
    for _site_root, _site_cells in _site_clusters.items():
        for _c in _site_cells:
            if _c not in cell_is_follower:
                _site_leader_groups[_site_root].append(_c)
    _rot_sites = [(k, v) for k, v in _site_leader_groups.items() if len(v) >= 2]

    _CAND_SAMPLE = 48   # candidates sampled per cell inside the target class
    rot_applied = 0
    for _rot_round in range(3):
        _round_gain = 0.0
        if progress_callback:
            progress_callback(96, f'mod3 site rotasyonu — tur {_rot_round + 1}/3')
        for _site_root, leaders in _rot_sites:
            base_classes = {l: assignment[l] % 3 for l in leaders}
            best_perm, best_delta, best_state = None, -1e-9, None
            for perm in _ROTATIONS:
                snapshot = {l: assignment[l] for l in leaders}
                delta = 0.0
                for l in leaders:
                    target_class = perm[base_classes[l]]
                    pool = pci_by_mod3[target_class]
                    if not pool:
                        continue
                    forbidden = _co_site_forbidden(l)
                    cands = random.sample(pool, min(_CAND_SAMPLE, len(pool)))
                    cands = [c for c in cands if c not in forbidden] or cands
                    # greedily take the lowest-energy candidate in the new class
                    cur = assignment[l]
                    best_c, best_ce = None, None
                    for c in cands:
                        ce = _group_energy(l, c, assignment)
                        if best_ce is None or ce < best_ce:
                            best_c, best_ce = c, ce
                    if best_c is not None and best_c != cur:
                        delta += _apply_pci(l, best_c)
                if delta < best_delta:
                    best_delta = delta
                    best_perm = perm
                    best_state = {l: assignment[l] for l in leaders}
                # revert to the pre-permutation state
                for l in leaders:
                    _apply_pci(l, snapshot[l])
            if best_perm is not None and best_state is not None:
                for l in leaders:
                    _apply_pci(l, best_state[l])
                _round_gain += best_delta
                rot_applied += 1
        if _round_gain > -1e-9:
            break   # no further improvement available

    if rot_applied:
        total_energy = 0.0
        for cid in independent_cells:
            total_energy += _group_energy(cid, assignment[cid], assignment)
        total_energy /= 2.0
        if total_energy < best_energy:
            best_energy = total_energy
            best_assignment = dict(assignment)
        else:
            # rotation pass did not help overall — fall back
            assignment = dict(best_assignment)
            pci_to_cells.clear()
            for cid in cell_ids:
                p = assignment.get(cid)
                if p is not None:
                    pci_to_cells[p].add(cid)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Post-SA: Greedy cleanup
    #   Phase 1: Fix remaining collisions (must be zero)
    #   Phase 2: Reduce confusions (swap PCIs if confusion count drops)
    # ------------------------------------------------------------------
    cleanup_count = 0
    for cid in independent_cells:
        pci_val = assignment[cid]
        # Check for collisions
        has_collision = False
        for nb_id in nb_excl_cosector.get(cid, set()):
            if assignment.get(nb_id) == pci_val:
                has_collision = True
                break
        if not has_collision:
            continue
        # Try to find collision-free PCI with best energy
        best_p = pci_val
        best_e = _group_energy(cid, pci_val, assignment)
        members = sector_members_cache[cid]
        # Constrained: only check PCIs from assigned mod3 class
        _cm3 = cell_mod3_class.get(cid)
        _pci_pool = pci_by_mod3[_cm3] if _cm3 is not None else range(max_pci)
        _co_fb_cleanup = _co_site_forbidden(cid)  # hard co-site exclusion
        for p in _pci_pool:
            if p == pci_val or p in _co_fb_cleanup:
                continue
            # Quick collision check
            collision_free = True
            for m in members:
                for nb_id in nb_excl_cosector.get(m, set()):
                    if assignment.get(nb_id) == p:
                        collision_free = False
                        break
                if not collision_free:
                    break
            if not collision_free:
                continue
            # Co-site mod3 check only for unconstrained cells (safety)
            if _cm3 is None:
                co_site_ok = True
                for m in members:
                    for nb_id in nb_excl_cosector.get(m, set()):
                        if (m, nb_id) in _co_site_flat:
                            if p % 3 == assignment.get(nb_id, -1) % 3:
                                co_site_ok = False
                                break
                    if not co_site_ok:
                        break
                if not co_site_ok:
                    continue
            # Evaluate energy (temp apply + revert with index)
            for m in members:
                pci_to_cells[pci_val].discard(m)
                assignment[m] = p
                pci_to_cells[p].add(m)
            e = _group_energy(cid, p, assignment)
            if e < best_e:
                best_e = e
                best_p = p
            # Revert for now
            for m in members:
                pci_to_cells[p].discard(m)
                assignment[m] = pci_val
                pci_to_cells[pci_val].add(m)
        # Apply best found + update reverse index
        if best_p != pci_val:
            for m in members:
                pci_to_cells[pci_val].discard(m)
                assignment[m] = best_p
                pci_to_cells[best_p].add(m)
            cleanup_count += 1

    # Phase 2: Confusion reduction sweep
    # For each independent cell with nonzero energy, try swapping to a PCI
    # that reduces confusion count without creating any new collisions.
    # Optimization: only test PCIs not used by any 1-hop neighbor (guaranteed
    # collision-free), sorted by least-used across 2-hop to minimize confusion.
    confusion_fixes = 0
    for cid in independent_cells:
        pci_val = assignment[cid]
        members = sector_members_cache[cid]
        cur_e = _group_energy(cid, pci_val, assignment)
        if cur_e <= 0:
            continue  # already perfect
        # Collect PCIs used by all neighbors of all members (blocked PCIs)
        blocked = set()
        for m in members:
            for nb_id in nb_excl_cosector.get(m, _EMPTY):
                nb_p = assignment.get(nb_id)
                if nb_p is not None:
                    blocked.add(nb_p)
        # Build candidate set: collision-free PCIs from assigned mod3 class
        _cm3 = cell_mod3_class.get(cid)
        _co_fb_cf = _co_site_forbidden(cid)  # hard co-site exclusion
        if _cm3 is not None:
            candidates = [p for p in pci_by_mod3[_cm3]
                          if p not in blocked and p not in _co_fb_cf]
        else:
            # Unconstrained: block co-site mod3 conflicts
            co_site_m3_blocked = set()
            for m in members:
                for nb_id in nb_excl_cosector.get(m, _EMPTY):
                    if (m, nb_id) in _co_site_flat:
                        nb_p = assignment.get(nb_id)
                        if nb_p is not None:
                            co_site_m3_blocked.add(nb_p % 3)
            candidates = [p for p in range(max_pci) if p not in blocked
                          and p % 3 not in co_site_m3_blocked
                          and p not in _co_fb_cf]
            if not candidates:
                candidates = [p for p in range(max_pci) if p not in blocked
                              and p not in _co_fb_cf]
        if not candidates:
            continue
        # Limit to reasonable number of candidates for performance
        if len(candidates) > 50:
            random.shuffle(candidates)
            candidates = candidates[:50]
        best_p = pci_val
        best_e = cur_e
        for p in candidates:
            if p == pci_val:
                continue
            # Evaluate energy improvement (temp apply + revert with index)
            for m in members:
                pci_to_cells[pci_val].discard(m)
                assignment[m] = p
                pci_to_cells[p].add(m)
            e = _group_energy(cid, p, assignment)
            if e < best_e:
                best_e = e
                best_p = p
            for m in members:
                pci_to_cells[p].discard(m)
                assignment[m] = pci_val
                pci_to_cells[pci_val].add(m)
        if best_p != pci_val:
            for m in members:
                pci_to_cells[pci_val].discard(m)
                assignment[m] = best_p
                pci_to_cells[best_p].add(m)
            confusion_fixes += 1

    # Phase 3: Site-level co-site mod3 cleanup — GUARANTEE zero violations
    # For each site that has co-site mod3 violations:
    #   1. Collect all sector-units at the site
    #   2. Build intra-site neighbor adjacency graph between units
    #   3. Graph-color with 3 colors (only adjacent units need different colors)
    #   4. Re-assign PCIs to match new mod3 classes
    cosite_m3_fixes = 0

    # Build follower → leader mapping
    _follower_to_leader = {}
    for _ldr in independent_cells:
        for _m in sector_members_cache[_ldr]:
            if _m != _ldr:
                _follower_to_leader[_m] = _ldr

    def _find_leader(cid):
        if cid in indep_idx:
            return cid
        return _follower_to_leader.get(cid)

    # Rebuild site clusters using the same Union-Find from Phase 0
    # (reuse _sfind which is still valid)
    _ph3_site_clusters = defaultdict(set)
    for cid in cell_ids:
        _ph3_site_clusters[_sfind(cid)].add(cid)

    for _site_root, _site_cells in _ph3_site_clusters.items():
        if len(_site_cells) <= 1:
            continue

        # Collect sector-units at this site (leader → members)
        _site_leaders = set()
        for cid in _site_cells:
            ldr = _find_leader(cid)
            if ldr is not None and ldr in indep_idx:
                _site_leaders.add(ldr)

        if len(_site_leaders) <= 1:
            continue

        # Filter to outdoor leaders only — indoor co-site mod3 is acceptable
        _leaders_list = sorted(
            ldr for ldr in _site_leaders
            if not all(m in _indoor_cells for m in sector_members_cache.get(ldr, [ldr]))
        )

        if len(_leaders_list) <= 1:
            continue

        # Check if this site has ANY co-site mod3 violation (outdoor only)
        has_violation = False
        for i, la in enumerate(_leaders_list):
            for lb in _leaders_list[i+1:]:
                pa, pb = assignment.get(la), assignment.get(lb)
                if pa is None or pb is None:
                    continue
                if pa % 3 != pb % 3:
                    continue
                # Check if any member of la is a co-site neighbor of any member of lb
                mem_a = sector_members_cache.get(la, [la])
                mem_b = sector_members_cache.get(lb, [lb])
                for ma in mem_a:
                    for mb in mem_b:
                        if (ma, mb) in _co_site_flat and mb in nb_excl_cosector.get(ma, set()):
                            has_violation = True
                            break
                    if has_violation:
                        break
                if has_violation:
                    break
            if has_violation:
                break

        if not has_violation:
            continue

        # Build intra-site adjacency: which leader-pairs are co-site neighbors?
        _adj = defaultdict(set)  # leader → set of adjacent leaders at same site
        for i, la in enumerate(_leaders_list):
            for lb in _leaders_list[i+1:]:
                connected = False
                mem_a = sector_members_cache.get(la, [la])
                mem_b = sector_members_cache.get(lb, [lb])
                for ma in mem_a:
                    for mb in mem_b:
                        if (ma, mb) in _co_site_flat and mb in nb_excl_cosector.get(ma, set()):
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    _adj[la].add(lb)
                    _adj[lb].add(la)

        # Graph-color with 3 colors using greedy + backtracking
        # Sort by degree descending (most constrained first)
        _sorted_leaders = sorted(_leaders_list, key=lambda x: -len(_adj.get(x, set())))
        _color = {}  # leader → mod3 class

        def _greedy_color():
            """Greedy graph coloring. Returns True if successful."""
            _color.clear()
            for ldr in _sorted_leaders:
                used = set()
                for adj_ldr in _adj.get(ldr, set()):
                    if adj_ldr in _color:
                        used.add(_color[adj_ldr])
                # Pick color that minimizes inter-site neighbor mod3 conflicts
                best_c = None
                best_cost = float('inf')
                for c in (0, 1, 2):
                    if c in used:
                        continue
                    cost = 0
                    for m in sector_members_cache.get(ldr, [ldr]):
                        for nb_id in nb_excl_cosector.get(m, set()):
                            if nb_id in _site_cells:
                                continue  # intra-site handled by graph coloring
                            nb_p = assignment.get(nb_id)
                            if nb_p is not None and nb_p % 3 == c:
                                cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_c = c
                if best_c is None:
                    # All 3 colors taken by adjacent leaders — shouldn't happen
                    # for planar site graphs, but use least-adjacent color
                    color_counts = {0: 0, 1: 0, 2: 0}
                    for adj_ldr in _adj.get(ldr, set()):
                        if adj_ldr in _color:
                            color_counts[_color[adj_ldr]] += 1
                    best_c = min(color_counts, key=color_counts.get)
                _color[ldr] = best_c
            return True

        _greedy_color()

        # Apply new mod3 classes: change PCIs of leaders whose current
        # mod3 doesn't match the assigned color
        for ldr in _sorted_leaders:
            target_m3 = _color[ldr]
            cur_p = assignment[ldr]
            if cur_p is not None and cur_p % 3 == target_m3:
                continue  # already correct

            members = sector_members_cache.get(ldr, [ldr])
            old_p = assignment[ldr]

            # Collect neighbor PCIs to avoid collisions
            nb_pcis = set()
            for m in members:
                for nb_id in nb_excl_cosector.get(m, set()):
                    nbp = assignment.get(nb_id)
                    if nbp is not None:
                        nb_pcis.add(nbp)
            # Hard co-site exclusion
            _co_fb_ph3 = _co_site_forbidden(ldr)

            # Find best PCI from target mod3 class
            pool = [p for p in pci_by_mod3[target_m3]
                    if p not in nb_pcis and p not in _co_fb_ph3 and p != old_p]
            if not pool:
                pool = [p for p in pci_by_mod3[target_m3]
                        if p not in _co_fb_ph3 and p != old_p]
            if not pool:
                pool = [p for p in pci_by_mod3[target_m3] if p != old_p]
            if not pool:
                continue

            best_p = None
            best_e = float('inf')
            test_pool = random.sample(pool, min(40, len(pool)))
            for p in test_pool:
                for m in members:
                    pci_to_cells[old_p].discard(m)
                    assignment[m] = p
                    pci_to_cells[p].add(m)
                e = _group_energy(ldr, p, assignment)
                if e < best_e:
                    best_e = e
                    best_p = p
                for m in members:
                    pci_to_cells[p].discard(m)
                    assignment[m] = old_p
                    pci_to_cells[old_p].add(m)

            if best_p is not None:
                for m in members:
                    pci_to_cells[old_p].discard(m)
                    assignment[m] = best_p
                    pci_to_cells[best_p].add(m)
                cell_mod3_class[ldr] = best_p % 3
                for m in members:
                    cell_mod3_class[m] = best_p % 3
                cosite_m3_fixes += 1

    # ------------------------------------------------------------------
    # Phase 4: Final co-site mod3 verification & forced fix
    # Uses the EXACT same detection logic as the scoring function
    # (_count_cosite / _cs_cnt) to guarantee zero co-site mod3 violations.
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(98, 'Faz 4: Co-site mod3 son doğrulama…')

    # Build site_id map for co-site detection (mirrors _cs_cnt in app.py)
    _sm_map = {}
    if 'site_id' in df.columns:
        for _, r in df.iterrows():
            sid = str(r.get('site_id', '')).strip()
            cid_s = str(r['cell_id'])
            if sid and sid != 'nan' and sid != '':
                _sm_map[cid_s] = sid

    def _is_cosite_exact(c1, c2):
        """Same-site check matching EXACTLY what scoring counts."""
        if _sm_map:
            s1, s2 = _sm_map.get(c1), _sm_map.get(c2)
            if s1 and s2 and s1 == s2 and s1 not in ('', 'nan'):
                return True
        return _is_same_site_by_id(c1, c2)

    phase4_fixes = 0
    for _ph4_pass in range(20):  # up to 20 iterative passes
        # Detect ALL co-site mod3 violations (same logic as scoring)
        _violations = []  # list of (cell_a, cell_b) pairs with co-site mod3
        _seen_ph4 = set()
        for cid_a in cell_ids:
            pa = assignment.get(cid_a)
            if pa is None:
                continue
            for nb_id in nb_excl_cosector.get(cid_a, set()):
                pb = assignment.get(nb_id)
                if pb is None:
                    continue
                pair = tuple(sorted([cid_a, nb_id]))
                if pair in _seen_ph4:
                    continue
                _seen_ph4.add(pair)
                if int(pa) % 3 != int(pb) % 3:
                    continue
                # Skip indoor cells — indoor co-site mod3 is acceptable
                if cid_a in _indoor_cells or nb_id in _indoor_cells:
                    continue
                # Check co-sector exclusion (same as _mod_conflict)
                if _is_co_sector_by_id(cid_a, nb_id):
                    continue
                sa_sec = cell_to_sector.get(cid_a)
                sb_sec = cell_to_sector.get(nb_id)
                if sa_sec and sa_sec == sb_sec:
                    continue
                # Check same-site (same as _cs_cnt)
                if _is_cosite_exact(cid_a, nb_id):
                    _violations.append(pair)

        if not _violations:
            break  # all clean

        # Fix each violation: change the leader with fewer constraints
        for va, vb in _violations:
            # Re-check (may have been fixed by earlier fix in this pass)
            pa, pb = assignment.get(va), assignment.get(vb)
            if pa is None or pb is None:
                continue
            if int(pa) % 3 != int(pb) % 3:
                continue
            if not _is_cosite_exact(va, vb):
                continue

            # Pick the cell to change: prefer the one with fewer neighbors
            ldr_a = _find_leader(va) if va not in indep_idx else va
            ldr_b = _find_leader(vb) if vb not in indep_idx else vb

            # If either is not an independent cell (can't change), pick the other
            candidates_to_change = []
            if ldr_a and ldr_a in indep_idx:
                candidates_to_change.append(ldr_a)
            if ldr_b and ldr_b in indep_idx:
                candidates_to_change.append(ldr_b)
            if not candidates_to_change:
                continue

            # Try changing each candidate; pick the one with least energy increase
            best_change = None  # (leader, new_pci, energy)
            for ldr in candidates_to_change:
                cur_p = assignment[ldr]
                cur_m3 = int(cur_p) % 3
                # Need a PCI with DIFFERENT mod3
                target_m3s = [m for m in (0, 1, 2) if m != cur_m3]
                members = sector_members_cache.get(ldr, [ldr])

                # Collect co-site neighbor mod3 classes to avoid
                cosite_nb_m3 = set()
                for m in members:
                    for nb_id in nb_excl_cosector.get(m, set()):
                        if _is_cosite_exact(m, nb_id):
                            nb_p = assignment.get(nb_id)
                            if nb_p is not None:
                                cosite_nb_m3.add(int(nb_p) % 3)

                # Pick target mod3 that doesn't conflict with other co-site neighbors
                valid_m3s = [m for m in target_m3s if m not in cosite_nb_m3]
                if not valid_m3s:
                    valid_m3s = target_m3s  # pigeonhole — pick least bad

                for tm3 in valid_m3s:
                    # Collect neighbor PCIs to avoid collisions
                    nb_pcis = set()
                    for m in members:
                        for nb_id in nb_excl_cosector.get(m, set()):
                            nbp = assignment.get(nb_id)
                            if nbp is not None:
                                nb_pcis.add(nbp)
                    # Hard co-site exclusion
                    _co_fb_ph4 = _co_site_forbidden(ldr)

                    pool = [p for p in pci_by_mod3[tm3]
                            if p not in nb_pcis and p not in _co_fb_ph4]
                    if not pool:
                        pool = [p for p in pci_by_mod3[tm3]
                                if p not in _co_fb_ph4]
                    if not pool:
                        pool = pci_by_mod3[tm3]
                    if not pool:
                        continue

                    test_pool = random.sample(pool, min(30, len(pool)))
                    old_p = assignment[ldr]
                    for p in test_pool:
                        for m in members:
                            pci_to_cells[old_p].discard(m)
                            assignment[m] = p
                            pci_to_cells[p].add(m)
                        e = _group_energy(ldr, p, assignment)
                        if best_change is None or e < best_change[2]:
                            best_change = (ldr, p, e)
                        for m in members:
                            pci_to_cells[p].discard(m)
                            assignment[m] = old_p
                            pci_to_cells[old_p].add(m)

            if best_change:
                ldr, new_p, _ = best_change
                old_p = assignment[ldr]
                members = sector_members_cache.get(ldr, [ldr])
                for m in members:
                    pci_to_cells[old_p].discard(m)
                    assignment[m] = new_p
                    pci_to_cells[new_p].add(m)
                cell_mod3_class[ldr] = new_p % 3
                for m in members:
                    cell_mod3_class[m] = new_p % 3
                phase4_fixes += 1

    # ------------------------------------------------------------------
    # Build result DataFrame
    # ------------------------------------------------------------------
    rows = []
    for cid in cell_ids:
        cur = pci_map.get(cid)
        cur_int = int(cur) if pd.notna(cur) else None
        planned = assignment.get(cid)
        changed = (planned != cur_int) if (planned is not None and cur_int is not None) else True
        pss_p, sss_p = decompose_pci(planned) if planned is not None else ('—', '—')
        pss_c, sss_c = decompose_pci(cur_int) if cur_int is not None else ('—', '—')

        if not changed:
            reason = 'Mevcut PCI uygun'
            level = 'SA — Korundu'
        elif planned is not None:
            reason = f"PCI {cur_int}→{planned} (mod3={planned%3}, mod6={planned%6}) [SA Optimized]"
            level = 'SA — Değiştirildi'
        else:
            reason = 'PCI atanamadı'
            level = 'Hata'

        rows.append({
            'cell_id': cid,
            'current_pci': cur_int if cur_int is not None else '—',
            'current_pss': pss_c, 'current_sss': sss_c,
            'planned_pci': planned if planned is not None else '—',
            'planned_pss': pss_p, 'planned_sss': sss_p,
            'changed': '✅ Değişti' if changed else '— Aynı',
            'relaxation_level': level,
            'reason': reason
        })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        assert_pci_range(result_df['planned_pci'], technology, 'plan_pci_network')
        for col in ['current_pci','planned_pci','current_pss','planned_pss','current_sss','planned_sss']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
    result_df = enrich_df_with_sector_info(result_df)
    return result_df
