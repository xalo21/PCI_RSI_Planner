"""Quick sanity test for PCI/RSI engine v2 + data handler v2."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from pci_engine import (
    run_full_analysis, build_neighbor_table, compute_cell_prach_info,
    cell_range_from_ncs, cell_range_from_format, roots_needed,
    get_ncs, get_lte_preamble_format, preambles_per_root,
    rsi_overlap, decompose_pci, haversine_distance,
    suggest_pci, suggest_rsi, plan_rsi_network, plan_pci_network,
    detect_sector_groups, _extract_sector_number, _extract_band_prefix,
    _is_co_sector_by_id, _MIXED_INDOOR_SITES, _DETECTED_INDOOR_CELLS,
    LTE_NCS_UNRESTRICTED, NZC_LONG,
    derive_zcz_from_cell_range, _effective_zcz
)
from data_handler import read_excel_file, generate_sample_excel, export_results_to_excel

def main():
    # ------ 3GPP table spot-checks ------
    assert get_ncs(0, 'LTE') == 0, "Ncs(0) should be 0"
    assert get_ncs(5, 'LTE') == 26, f"Ncs(5)={get_ncs(5,'LTE')}, expected 26"
    assert get_ncs(15, 'LTE') == 419, f"Ncs(15)={get_ncs(15,'LTE')}, expected 419"
    # FDD, TS 36.211 Table 5.7.1-2: 0-15→0, 16-31→1, 32-47→2, 48-63→3.
    # v2 asserted get_lte_preamble_format(60) == 4 — that was the K-3 bug being
    # locked in by its own test.  Format 4 does not exist in FDD; it lives in
    # the TDD table at indices 48-57, where 58-63 are N/A.
    assert get_lte_preamble_format(0) == 0
    assert get_lte_preamble_format(20) == 1
    assert get_lte_preamble_format(40) == 2
    assert get_lte_preamble_format(50) == 3
    assert get_lte_preamble_format(60) == 3, "FDD'de format 4 yok"
    assert get_lte_preamble_format(50, 'TDD') == 4
    assert get_lte_preamble_format(60, 'TDD') is None, "TDD'de 58-63 N/A"
    print("✅ 3GPP table spot-checks passed")

    # ------ Cell range ------
    cr = cell_range_from_ncs(26)
    assert 1.0 < cr < 20.0, f"cell_range_from_ncs(26)={cr} km – unexpected"
    fmt_cr = cell_range_from_format(0)
    assert fmt_cr > 0
    print(f"✅ Cell range Ncs=26 → {cr:.2f} km, Format 0 → {fmt_cr:.2f} km")

    # ------ Roots needed ------
    ppr = preambles_per_root(26)
    assert ppr == 32, f"preambles_per_root(26)={ppr}, expected 32"
    rn = roots_needed(64, 26)
    assert rn == 2, f"roots_needed(64,26)={rn}, expected 2"
    print(f"✅ Ncs=26 → {ppr} preambles/root, {rn} roots needed")

    # ------ RSI overlap ------
    assert rsi_overlap(42, 26, 43, 26) == True, "RSI 42 & 43 should overlap (Ncs=26)"
    assert rsi_overlap(42, 26, 45, 26) == False, "RSI 42 & 45 should not overlap (Ncs=26)"
    print("✅ RSI overlap detection passed")

    # ------ Sample Excel round-trip ------
    sample_buf = generate_sample_excel()
    df, msgs = read_excel_file(sample_buf)
    assert df is not None and len(df) == 18, f"Expected 18 cells, got {len(df) if df is not None else 0}"
    assert 'prach_config_index' in df.columns
    assert 'zero_correlation_zone' in df.columns
    print(f"✅ Sample Excel: {len(df)} cells, columns OK")

    # ------ Full analysis ------
    results = run_full_analysis(df, radius_km=3.0, technology='LTE',
                                use_antenna_direction=True, default_beamwidth=65.0,
                                include_intra_site=True)
    s = results['summary']
    print(f"\n--- Analysis Summary ---")
    print(f"  Cells: {s['total_cells']}")
    print(f"  Neighbor pairs: {s['total_neighbor_pairs']}")
    print(f"  Neighbor sources: {s.get('neighbor_sources', {})}")
    print(f"  Collisions: {s['collision_count']}")
    print(f"  Confusions: {s['confusion_count']}")
    print(f"  Mod3: {s['mod3_conflict_count']}")
    print(f"  RSI collisions: {s['rsi_collision_count']}")
    print(f"  Health score: {s['health_score']}")

    # ------ Neighbor table & export ------
    nt = build_neighbor_table(df, results['neighbors'], results.get('neighbor_sources'),
                               nbr_attempts=results.get('neighbor_attempts'))
    prach_rows = [compute_cell_prach_info(row) for _, row in df.iterrows()]
    prach_df = pd.DataFrame(prach_rows)
    xl = export_results_to_excel(df, results, nt, prach_df)
    assert len(xl) > 1000, "Export file too small"
    print(f"✅ Export Excel: {len(xl)} bytes")

    # ------ PCI decompose ------
    pss, sss = decompose_pci(100)
    assert pss == 1 and sss == 33, f"decompose_pci(100) = ({pss},{sss})"
    print("✅ decompose_pci passed")

    # ------ PCI suggestions ------
    pci_sug = suggest_pci(df, results['neighbors'], results, 'LTE', True, True, False,
                          nbr_attempts=results.get('neighbor_attempts'))
    print(f"✅ PCI suggestions: {len(pci_sug)} recommendations")
    if len(pci_sug) > 0:
        for _, r in pci_sug.iterrows():
            print(f"   {r['cell_id']}: PCI {r['current_pci']} → {r['suggested_pci']}  ({r['issues']})")

    # ------ RSI suggestions ------
    rsi_sug = suggest_rsi(df, results['neighbors'], results, 'LTE')
    print(f"✅ RSI suggestions: {len(rsi_sug)} recommendations")
    if len(rsi_sug) > 0:
        for _, r in rsi_sug.iterrows():
            print(f"   {r['cell_id']}: RSI {r['current_rsi']} → {r['suggested_rsi']}  (conflicts: {r['conflicting_with']})")

    # ------ Full network RSI plan ------
    rsi_plan = plan_rsi_network(df, results['neighbors'], 'LTE')
    assert len(rsi_plan) == len(df), f"RSI plan has {len(rsi_plan)} rows, expected {len(df)}"
    planned_ok = rsi_plan[rsi_plan['planned_rsi'] != '—']
    print(f"✅ RSI auto-plan: {len(rsi_plan)} cells, {len(planned_ok)} assigned")
    for _, r in rsi_plan.iterrows():
        print(f"   {r['cell_id']}: RSI {r['current_rsi']} → {r['planned_rsi']} "
              f"({r['roots_needed']} roots, range={r['planned_range']}, cr={r['cell_range_km']}km) [{r['changed']}]")

    # Verify no RSI overlap in planned values
    assigned_map = {}
    ncs_map = {}
    for _, r in rsi_plan.iterrows():
        if r['planned_rsi'] != '—':
            assigned_map[r['cell_id']] = int(r['planned_rsi'])
            ncs_map[r['cell_id']] = int(r['ncs'])
    overlap_count = 0
    for c, ns in results['neighbors'].items():
        c = str(c)
        if c not in assigned_map: continue
        for nb in ns:
            nb = str(nb)
            if nb not in assigned_map: continue
            if c >= nb: continue  # check each pair once
            if rsi_overlap(assigned_map[c], ncs_map[c], assigned_map[nb], ncs_map[nb]):
                overlap_count += 1
                print(f"   ⚠️ OVERLAP: {c} RSI={assigned_map[c]} vs {nb} RSI={assigned_map[nb]}")
    assert overlap_count == 0, f"RSI plan has {overlap_count} overlaps!"
    print(f"✅ RSI plan verification: 0 overlaps among neighbors")

    # ------ Full network PCI plan ------
    pci_plan = plan_pci_network(df, results['neighbors'], 'LTE', True, True, False,
                                nbr_attempts=results.get('neighbor_attempts'))
    assert len(pci_plan) == len(df), f"PCI plan has {len(pci_plan)} rows, expected {len(df)}"
    print(f"✅ PCI auto-plan: {len(pci_plan)} cells")
    for _, r in pci_plan.iterrows():
        print(f"   {r['cell_id']}: PCI {r['current_pci']} → {r['planned_pci']} "
              f"(PSS={r['planned_pss']}, SSS={r['planned_sss']}) [{r['changed']}]")

    # ------ Verify co-site mod3 uniqueness ------
    # All cells on the same site must have different mod3 (PSS) values
    site_map = dict(zip(df['cell_id'].astype(str), df['site_id'].astype(str))) if 'site_id' in df.columns else {}
    plan_pci_map = {}
    for _, r in pci_plan.iterrows():
        if r['planned_pci'] != '—':
            plan_pci_map[str(r['cell_id'])] = int(r['planned_pci'])
    from collections import defaultdict as dd
    site_mod3 = dd(list)
    for cid, pci in plan_pci_map.items():
        sid = site_map.get(cid, cid)
        site_mod3[sid].append((cid, pci, pci % 3))
    co_site_mod3_violations = 0
    co_site_collision_violations = 0
    for sid, cells in site_mod3.items():
        mod3s = [m for _, _, m in cells]
        pcis = [p for _, p, _ in cells]
        if len(mod3s) != len(set(mod3s)):
            co_site_mod3_violations += 1
            print(f"   ⚠️ Co-site mod3 violation at {sid}: {cells}")
        if len(pcis) != len(set(pcis)):
            co_site_collision_violations += 1
            print(f"   ⚠️ Co-site PCI collision at {sid}: {cells}")
    assert co_site_mod3_violations == 0, f"{co_site_mod3_violations} co-site mod3 violations!"
    assert co_site_collision_violations == 0, f"{co_site_collision_violations} co-site PCI collisions!"
    print(f"✅ Co-site mod3 verification: 0 violations across {len(site_mod3)} sites")
    print(f"✅ Co-site collision verification: 0 collisions across {len(site_mod3)} sites")

    # ------ Reserved PCI range test ------
    from pci_engine import detect_sector_groups as _dsg
    sg, c2s = _dsg(df)
    pci_plan_reserved = plan_pci_network(df, results['neighbors'], 'LTE',
                                         sector_groups=sg, cell_to_sector=c2s,
                                         reserved_pci_start=0, reserved_pci_end=99)
    for _, r in pci_plan_reserved.iterrows():
        if str(r['planned_pci']) not in ('—', 'nan', 'None'):
            p = int(r['planned_pci'])
            assert p < 0 or p > 99, f"PCI {p} is in reserved range 0-99!"
    print("✅ Reserved PCI range: no planned PCI in 0-99 range")

    # ------ SA iterations override test ------
    pci_plan_fast = plan_pci_network(df, results['neighbors'], 'LTE',
                                     sector_groups=sg, cell_to_sector=c2s,
                                     sa_iterations_override=10000)
    assert len(pci_plan_fast) == 18, f"Expected 18 cells, got {len(pci_plan_fast)}"
    print("✅ SA iterations override: 10K iterations completed successfully")

    # ------ New cell PCI/RSI finder ------
    from pci_engine import find_optimal_pci_rsi_for_new_cells
    new_cell = pd.DataFrame([{
        'cell_id': 'NEW_001', 'site_id': 'NEWSITE',
        'latitude': df.iloc[0]['latitude'] + 0.005,  # close to SITE001
        'longitude': df.iloc[0]['longitude'] + 0.005,
        'azimuth': 0, 'beamwidth': 65,
        'pci': None, 'rsi': None,
        'zero_correlation_zone': 5, 'prach_config_index': 0, 'earfcn': 0,
    }])
    nc_result = find_optimal_pci_rsi_for_new_cells(df, new_cell, 3.0, 'LTE')
    assert len(nc_result) == 1, f"Expected 1 result, got {len(nc_result)}"
    r = nc_result.iloc[0]
    assert r['suggested_pci'] != '—', "Should find a PCI for new cell"
    assert r['suggested_rsi'] != '—', "Should find an RSI for new cell"
    assert r['neighbors_found'] > 0, "New cell should have neighbours"
    print(f"✅ New cell finder: {r['cell_id']} → PCI={r['suggested_pci']} "
          f"(PSS={r['pss']},SSS={r['sss']}), RSI={r['suggested_rsi']} "
          f"({r['roots_needed']} roots, range={r['rsi_range']}), "
          f"{r['neighbors_found']} neighbors [{r['pci_quality']}]")

    # ── Test: 3-Rule co-sector detection ──

    # ── Scenario 1: Unbalanced site AS001 → RULE 2 (azimuth fallback) ──
    #   Naming sector 1 (A): L,Z,T = 3 bands; sector 2 (D): L,Z,T = 3 bands
    #   Naming sector 3 (G): L,Z = 2 bands → UNEQUAL → Rule 2
    #   Azimuth grouping: 0°={L..A,Z..A}, 120°={L..D,Z..D,T..A}, 240°={L..G,Z..G,T..D}
    #   TAS001A (T, sec 1, az 120°) — sec 1 has {L,Z,T} → NOT 1800-only → NOT indoor
    mixed_rows = [
        {'cell_id':'LAS001A','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':0,'beamwidth':65,'pci':1,'rsi':0,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS001A','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':0,'beamwidth':65,'pci':1,'rsi':0,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'LAS001D','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':120,'beamwidth':65,'pci':2,'rsi':10,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS001D','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':120,'beamwidth':65,'pci':2,'rsi':10,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'LAS001G','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':240,'beamwidth':65,'pci':3,'rsi':20,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS001G','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':240,'beamwidth':65,'pci':3,'rsi':20,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        # Indoor cells: same naming letters A,D but azimuths differ from outdoor A,D
        {'cell_id':'TAS001A','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':120,'beamwidth':65,'pci':2,'rsi':10,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS001D','site_id':'S001','latitude':40.0,'longitude':29.0,'azimuth':240,'beamwidth':65,'pci':3,'rsi':20,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
    ]
    df_mixed = pd.DataFrame(mixed_rows)
    sg, c2s = detect_sector_groups(df_mixed)

    # Verify AS001 is detected as mixed (Rule 2)
    assert 'AS001' in _MIXED_INDOOR_SITES, \
        f"AS001 should be in _MIXED_INDOOR_SITES, got {_MIXED_INDOOR_SITES}"

    # _is_co_sector_by_id should return False for mixed site → callers use cell_to_sector
    assert not _is_co_sector_by_id('TAS001A', 'ZAS001A'), \
        "Rule 2 site: _is_co_sector_by_id must return False (naming unreliable)"
    assert not _is_co_sector_by_id('LAS001A', 'ZAS001A'), \
        "Rule 2 site: _is_co_sector_by_id must return False even for same-band"

    # cell_to_sector should correctly group by azimuth:
    assert c2s.get('TAS001A') == c2s.get('LAS001D'), \
        f"TAS001A and LAS001D should be co-sector (azimuth 120°): {c2s.get('TAS001A')} vs {c2s.get('LAS001D')}"
    assert c2s.get('TAS001A') == c2s.get('ZAS001D'), \
        f"TAS001A and ZAS001D should be co-sector (azimuth 120°)"
    assert c2s.get('TAS001A') != c2s.get('ZAS001A'), \
        f"TAS001A (120°) and ZAS001A (0°) should NOT be co-sector"
    assert c2s.get('LAS001A') == c2s.get('ZAS001A'), \
        f"LAS001A and ZAS001A should be co-sector (azimuth 0°)"
    # TAS001A is NOT 1800-only (sector 1 has L,Z,T) → NOT indoor
    assert 'TAS001A' not in _DETECTED_INDOOR_CELLS, \
        "TAS001A is not 1800-only in its naming sector → should NOT be indoor"

    print("✅ Rule 2: unbalanced site AS001 uses azimuth fallback correctly")

    # ── Scenario 2: Balanced site AS002 → RULE 1 (naming convention) ──
    #   3 sectors × 3 bands each, azimuths within each sector < 10°
    balanced_rows = [
        {'cell_id':'LAS002A','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':0,'beamwidth':65,'pci':10,'rsi':0,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS002A','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':2,'beamwidth':65,'pci':10,'rsi':0,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS002A','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':358,'beamwidth':65,'pci':10,'rsi':0,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'LAS002D','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':120,'beamwidth':65,'pci':11,'rsi':10,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS002D','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':122,'beamwidth':65,'pci':11,'rsi':10,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS002D','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':118,'beamwidth':65,'pci':11,'rsi':10,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'LAS002G','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':240,'beamwidth':65,'pci':12,'rsi':20,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS002G','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':242,'beamwidth':65,'pci':12,'rsi':20,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS002G','site_id':'S002','latitude':41.0,'longitude':30.0,'azimuth':238,'beamwidth':65,'pci':12,'rsi':20,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
    ]
    df_bal = pd.DataFrame(balanced_rows)
    sg2, c2s2 = detect_sector_groups(df_bal)

    # Balanced + azimuth consistent → RULE 1 → naming convention
    assert 'AS002' not in _MIXED_INDOOR_SITES, \
        f"AS002 should NOT be in _MIXED_INDOOR_SITES (Rule 1 balanced site)"
    assert _is_co_sector_by_id('LAS002A', 'TAS002A'), \
        "Rule 1: naming convention should be used for balanced site"
    assert c2s2.get('LAS002A') == c2s2.get('TAS002A'), \
        f"LAS002A and TAS002A should be co-sector (naming convention)"
    assert c2s2.get('LAS002A') != c2s2.get('LAS002D'), \
        f"LAS002A and LAS002D should NOT be co-sector (different sectors)"

    print("✅ Rule 1: balanced site AS002 uses naming convention correctly")

    # ── Scenario 3: Pure indoor site AS003 → RULE 3 ──
    #   All cells are 1800-only (T-band) AND all share same azimuth (0°)
    #   → All cells are indoor, naming convention for sector grouping
    indoor_rows = [
        {'cell_id':'TAS003A','site_id':'S003','latitude':42.0,'longitude':31.0,'azimuth':0,'beamwidth':65,'pci':20,'rsi':0,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS003B','site_id':'S003','latitude':42.0,'longitude':31.0,'azimuth':0,'beamwidth':65,'pci':20,'rsi':0,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS003D','site_id':'S003','latitude':42.0,'longitude':31.0,'azimuth':0,'beamwidth':65,'pci':21,'rsi':10,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS003E','site_id':'S003','latitude':42.0,'longitude':31.0,'azimuth':0,'beamwidth':65,'pci':21,'rsi':10,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
    ]
    df_indoor = pd.DataFrame(indoor_rows)
    sg3, c2s3 = detect_sector_groups(df_indoor)

    # Pure indoor: should NOT be mixed
    assert 'AS003' not in _MIXED_INDOOR_SITES, \
        f"AS003 should NOT be in _MIXED_INDOOR_SITES (Rule 3 pure indoor)"
    # Rule 3: ALL cells should be detected as indoor
    assert 'TAS003A' in _DETECTED_INDOOR_CELLS, "TAS003A should be detected indoor (Rule 3)"
    assert 'TAS003D' in _DETECTED_INDOOR_CELLS, "TAS003D should be detected indoor (Rule 3)"
    # Naming convention grouping still works
    assert _is_co_sector_by_id('TAS003A', 'TAS003B'), \
        "Rule 3: TAS003A and TAS003B should be co-sector (naming)"
    assert not _is_co_sector_by_id('TAS003A', 'TAS003D'), \
        "Rule 3: TAS003A and TAS003D should NOT be co-sector"
    assert c2s3.get('TAS003A') != c2s3.get('TAS003D'), \
        f"TAS003A and TAS003D should be different sectors"
    assert c2s3.get('TAS003A') == c2s3.get('TAS003B'), \
        f"TAS003A and TAS003B should be same sector"

    print("✅ Rule 3: pure indoor site AS003 — all cells indoor, naming convention grouping")

    # ── Scenario 4: AS1702-style — RULE 2 (unbalanced bands) ──
    #   Outdoor sectors 1,2,3 have L,Z,E,T (4 bands each)
    #   Indoor sectors 4,5 have only T (1 band each)
    #   Band counts: [4,4,4,1,1] → UNEQUAL → Rule 2 → azimuth fallback
    #   TAS1702J (T-only, sec 4, az 0°) → indoor; TAS1702M (T-only, sec 5, az 0°) → indoor
    #   Indoor cells grouped by naming convention sector; outdoor by azimuth
    as1702_rows = [
        # Outdoor sector 1 (azimuth 0°)
        {'cell_id':'LAS1702A','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':0,'beamwidth':65,'pci':100,'rsi':0,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS1702A','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':0,'beamwidth':65,'pci':100,'rsi':0,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'EAS1702A','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':0,'beamwidth':65,'pci':100,'rsi':0,'earfcn':400,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS1702A','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':0,'beamwidth':65,'pci':100,'rsi':0,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        # Outdoor sector 2 (azimuth 120°)
        {'cell_id':'LAS1702D','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':120,'beamwidth':65,'pci':101,'rsi':10,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS1702D','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':120,'beamwidth':65,'pci':101,'rsi':10,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'EAS1702D','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':120,'beamwidth':65,'pci':101,'rsi':10,'earfcn':400,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS1702D','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':120,'beamwidth':65,'pci':101,'rsi':10,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        # Outdoor sector 3 (azimuth 240°)
        {'cell_id':'LAS1702G','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':240,'beamwidth':65,'pci':102,'rsi':20,'earfcn':100,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'ZAS1702G','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':240,'beamwidth':65,'pci':102,'rsi':20,'earfcn':200,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'EAS1702G','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':240,'beamwidth':65,'pci':102,'rsi':20,'earfcn':400,'zero_correlation_zone':5,'prach_config_index':0},
        {'cell_id':'TAS1702G','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':240,'beamwidth':65,'pci':102,'rsi':20,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        # Indoor sector 4 (azimuth 0° — T-only in sector 4)
        {'cell_id':'TAS1702J','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':0,'beamwidth':65,'pci':103,'rsi':30,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
        # Indoor sector 5 (azimuth 0° — T-only in sector 5)
        {'cell_id':'TAS1702M','site_id':'S1702','latitude':40.5,'longitude':29.5,'azimuth':0,'beamwidth':65,'pci':104,'rsi':40,'earfcn':300,'zero_correlation_zone':5,'prach_config_index':0},
    ]
    df_1702 = pd.DataFrame(as1702_rows)
    sg4, c2s4 = detect_sector_groups(df_1702)

    # AS1702 is UNBALANCED (band counts differ) → Rule 2 → mixed site
    assert 'AS1702' in _MIXED_INDOOR_SITES, \
        f"AS1702 should be in _MIXED_INDOOR_SITES (unbalanced band counts)"

    # _is_co_sector_by_id returns False for all Rule 2 sites
    assert not _is_co_sector_by_id('LAS1702A', 'TAS1702A'), \
        "Rule 2: _is_co_sector_by_id returns False for mixed site"

    # Outdoor 0° cells share azimuth group
    assert c2s4.get('LAS1702A') == c2s4.get('ZAS1702A'), \
        f"LAS1702A and ZAS1702A should be co-sector (azimuth 0°)"
    assert c2s4.get('LAS1702A') == c2s4.get('TAS1702A'), \
        f"LAS1702A and TAS1702A should be co-sector (outdoor sector 1)"
    # Indoor cells get SEPARATE groups (naming convention sector)
    assert c2s4.get('LAS1702A') != c2s4.get('TAS1702J'), \
        f"TAS1702J (indoor sec 4) must NOT share group with outdoor sec 1"
    assert c2s4.get('LAS1702A') != c2s4.get('TAS1702M'), \
        f"TAS1702M (indoor sec 5) must NOT share group with outdoor sec 1"
    assert c2s4.get('TAS1702J') != c2s4.get('TAS1702M'), \
        f"TAS1702J (sec 4) and TAS1702M (sec 5) must be separate sectors"
    # 120° and 240° groups separate
    assert c2s4.get('LAS1702D') == c2s4.get('TAS1702D'), \
        f"LAS1702D and TAS1702D should be co-sector (azimuth 120°)"
    assert c2s4.get('LAS1702A') != c2s4.get('LAS1702D'), \
        f"0° group and 120° group should NOT be co-sector"

    # Indoor detection: T-only sectors at azimuth 0° → indoor
    assert 'TAS1702J' in _DETECTED_INDOOR_CELLS, \
        "TAS1702J should be indoor (T-only in sector 4 + azimuth 0°)"
    assert 'TAS1702M' in _DETECTED_INDOOR_CELLS, \
        "TAS1702M should be indoor (T-only in sector 5 + azimuth 0°)"
    # TAS1702A is NOT indoor (sector 1 has L,Z,E,T → not 1800-only)
    assert 'TAS1702A' not in _DETECTED_INDOOR_CELLS, \
        "TAS1702A is not 1800-only in sector 1 → should NOT be indoor"

    print(f"✅ Rule 2: AS1702-style unbalanced site — indoor NC grouping + azimuth fallback correct")

    # ── Scenario 5: AS1702 PCI plan — indoor sectors get separate PCIs ──
    results_1702 = run_full_analysis(df_1702, radius_km=5.0)
    pci_plan_1702 = plan_pci_network(df_1702, results_1702['neighbors'], 'LTE',
                                      True, True, False,
                                      sector_groups=sg4, cell_to_sector=c2s4,
                                      sa_iterations_override=50000)
    _1702_pci = {}
    for _, r in pci_plan_1702.iterrows():
        if str(r['planned_pci']) not in ('—', 'nan', 'None'):
            _1702_pci[str(r['cell_id'])] = int(r['planned_pci'])
    # Indoor sector 4 (TAS1702J) must have different PCI from outdoor sector 1 (TAS1702A)
    _pci_j = _1702_pci.get('TAS1702J')
    _pci_a = _1702_pci.get('TAS1702A')
    _pci_m = _1702_pci.get('TAS1702M')
    _pci_d = _1702_pci.get('TAS1702D')  # outdoor sector 2
    _pci_g = _1702_pci.get('TAS1702G')  # outdoor sector 3
    assert _pci_j != _pci_a, \
        f"TAS1702J (indoor sec 4) PCI={_pci_j} must differ from TAS1702A (outdoor sec 1) PCI={_pci_a}"
    assert _pci_m != _pci_a, \
        f"TAS1702M (indoor sec 5) PCI={_pci_m} must differ from TAS1702A (outdoor sec 1) PCI={_pci_a}"
    assert _pci_j != _pci_m, \
        f"TAS1702J (sec 4) PCI={_pci_j} must differ from TAS1702M (sec 5) PCI={_pci_m}"
    # Co-sector cells within the same sector share PCI (correct), but
    # different sector groups must have distinct PCIs (no co-site collision)
    _sector_pcis = {_pci_a, _pci_d, _pci_g, _pci_j, _pci_m}
    assert len(_sector_pcis) == 5, \
        f"All 5 sectors must have unique PCIs: sec1={_pci_a}, sec2={_pci_d}, sec3={_pci_g}, sec4={_pci_j}, sec5={_pci_m}"
    print(f"✅ AS1702 PCI plan: all 5 sectors have unique PCIs (sec1={_pci_a}, sec2={_pci_d}, sec3={_pci_g}, sec4={_pci_j}, sec5={_pci_m})")

    # ------ sector column override test ------
    # When sector column is populated, it overrides naming convention
    df_sec = pd.DataFrame({
        'cell_id': ['EAS001A', 'TAS001A', 'LAS001A',  # same NC sector 1 → but sector col differs
                     'EAS001D', 'TAS001D'],
        'site_id': ['S001']*5,
        'latitude': [41.0]*5,
        'longitude': [29.0]*5,
        'azimuth': [0, 0, 120, 120, 120],
        'pci': [100, 100, 200, 200, 200],
        'sector': [1, 1, 2, 3, 3],  # TAS001A→sec1 with EAS001A, LAS001A→sec2 alone, EAS001D+TAS001D→sec3
        'technology': ['LTE']*5,
    })
    sg_sec, c2s_sec = detect_sector_groups(df_sec)
    # EAS001A and TAS001A should be in the same group (sector=1)
    assert c2s_sec['EAS001A'] == c2s_sec['TAS001A'], \
        f"sector col: EAS001A and TAS001A should share sector (both sector=1)"
    # LAS001A should be alone (sector=2) → different from EAS001A
    assert c2s_sec['LAS001A'] != c2s_sec['EAS001A'], \
        f"sector col: LAS001A (sector=2) should differ from EAS001A (sector=1)"
    # EAS001D and TAS001D should share (sector=3)
    assert c2s_sec['EAS001D'] == c2s_sec['TAS001D'], \
        f"sector col: EAS001D and TAS001D should share sector (both sector=3)"
    # sector=3 should differ from sector=1
    assert c2s_sec['EAS001D'] != c2s_sec['EAS001A'], \
        f"sector col: sector 3 should differ from sector 1"
    print("✅ sector column override: grouping matches sector column values")

    # ------ Huawei cellRange → zcz reverse mapping ------
    # 14500m → should pick zcz=12 (Ncs=119, range=17.02km ≥ 14.5km)
    zcz_14k, ncs_14k = derive_zcz_from_cell_range(14500)
    assert zcz_14k == 12 and ncs_14k == 119, f"14500m: zcz={zcz_14k}, ncs={ncs_14k}"
    # 38000m → should pick zcz=14 (Ncs=279, range=39.89km ≥ 38km)
    zcz_38k, ncs_38k = derive_zcz_from_cell_range(38000)
    assert zcz_38k == 14 and ncs_38k == 279, f"38000m: zcz={zcz_38k}, ncs={ncs_38k}"
    # 3000m → should pick zcz=4 (Ncs=22, range=3.15km ≥ 3km)
    zcz_3k, ncs_3k = derive_zcz_from_cell_range(3000)
    assert zcz_3k == 4 and ncs_3k == 22, f"3000m: zcz={zcz_3k}, ncs={ncs_3k}"
    # 0m → safe default (zcz=5, ncs=26)
    zcz_0, ncs_0 = derive_zcz_from_cell_range(0)
    assert zcz_0 == 5 and ncs_0 == 26, f"0m: zcz={zcz_0}, ncs={ncs_0}"
    print("✅ Huawei cellRange → zcz reverse mapping passed")

    # ------ _effective_zcz: cellRange priority ------
    row_huawei = {'cell_range': 14500, 'zero_correlation_zone': 5, 'prach_config_index': 0}
    assert _effective_zcz(row_huawei) == 12, "cell_range should override default zcz"
    row_nokia = {'zero_correlation_zone': 8, 'prach_config_index': 0}
    assert _effective_zcz(row_nokia) == 8, "explicit zcz should be used when no cell_range"
    row_no_cr = {'cell_range': None, 'zero_correlation_zone': 11}
    assert _effective_zcz(row_no_cr) == 11, "null cell_range should fall back to zcz"
    print("✅ _effective_zcz priority logic passed")

    # ------ compute_cell_prach_info with cellRange ------
    info_hw = compute_cell_prach_info({'cell_range': 14500, 'prach_config_index': 0,
                                        'zero_correlation_zone': 5})
    assert info_hw['ncs'] == 119, f"PRACH info with cellRange: ncs={info_hw['ncs']}, expected 119"
    assert info_hw['roots_needed'] == 10, f"roots_needed={info_hw['roots_needed']}, expected 10"
    print(f"✅ compute_cell_prach_info(cellRange=14500m): Ncs={info_hw['ncs']}, "
          f"roots={info_hw['roots_needed']}, range={info_hw['cell_range_ncs_km']}km")

    # ------ RSI plan with cellRange data ------
    cr_data = {
        'cell_id': [f'HW_SITE{s:02d}_{c}' for s in range(1, 4) for c in range(1, 4)],
        'site_id': [f'HW_SITE{s:02d}' for s in range(1, 4) for _ in range(3)],
        'latitude': [39.9 + s * 0.01 for s in range(1, 4) for _ in range(3)],
        'longitude': [32.8 + s * 0.01 for s in range(1, 4) for _ in range(3)],
        'azimuth': [0, 120, 240] * 3,
        'pci': list(range(9)),
        'rsi': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'cell_range': [14500, 14500, 14500, 29500, 29500, 29500, 3000, 3000, 3000],
        'prach_config_index': [0] * 9,
        'zero_correlation_zone': [5] * 9,  # default — should be overridden by cell_range
        'technology': ['LTE'] * 9,
        'beamwidth': [65] * 9,
    }
    hw_df = pd.DataFrame(cr_data)
    from pci_engine import find_neighbors
    hw_nb, _, _ = find_neighbors(hw_df, 5.0, True)
    hw_sg, hw_c2s = detect_sector_groups(hw_df)
    hw_plan = plan_rsi_network(hw_df, hw_nb, 'LTE', hw_sg, hw_c2s)
    assert len(hw_plan) == 9, f"Huawei RSI plan: {len(hw_plan)} rows"
    # Verify that cells with 14500m range have larger roots_needed than 3000m cells
    hw_roots = dict(zip(hw_plan['cell_id'], hw_plan['roots_needed']))
    assert hw_roots['HW_SITE01_1'] > hw_roots['HW_SITE03_1'], \
        f"14500m cell should need more roots ({hw_roots['HW_SITE01_1']}) than 3000m ({hw_roots['HW_SITE03_1']})"
    print(f"✅ Huawei cellRange RSI plan: 9 cells planned, roots verified")

    # ------ NR cellRange test ------
    # NR long sequence (prach_config=0): same Nzc/Tseq as LTE → same zcz mapping
    zcz_nr_long, ncs_nr_long = derive_zcz_from_cell_range(14500, technology='NR', preamble_format=0)
    assert zcz_nr_long == 12 and ncs_nr_long == 119, f"NR long 14500m: zcz={zcz_nr_long}, ncs={ncs_nr_long}"
    # NR short sequence (prach_config=30): L=139, Tseq=133.33µs → much smaller cell range per Ncs
    zcz_nr_short, ncs_nr_short = derive_zcz_from_cell_range(500, technology='NR', preamble_format=30)
    cr_short_km = cell_range_from_ncs(ncs_nr_short, 139, 133.33)
    assert cr_short_km >= 0.5, f"NR short 500m: zcz={zcz_nr_short}, ncs={ncs_nr_short}, range={cr_short_km}km"
    print(f"✅ NR cellRange: long(14500m)→zcz={zcz_nr_long}/Ncs={ncs_nr_long}, "
          f"short(500m)→zcz={zcz_nr_short}/Ncs={ncs_nr_short}")

    # NR cellRange RSI plan
    nr_cr_data = {
        'cell_id': [f'NR_HW_{s}_{c}' for s in range(1, 3) for c in range(1, 4)],
        'site_id': [f'NR_HW_{s}' for s in range(1, 3) for _ in range(3)],
        'latitude': [39.9 + s * 0.01 for s in range(1, 3) for _ in range(3)],
        'longitude': [32.8 + s * 0.01 for s in range(1, 3) for _ in range(3)],
        'azimuth': [0, 120, 240] * 2,
        'pci': list(range(6)),
        'rsi': [0] * 6,
        'cell_range': [14500, 14500, 14500, 3000, 3000, 3000],
        'prach_config_index': [0] * 6,  # NR long sequence
        'zero_correlation_zone': [5] * 6,
        'technology': ['NR'] * 6,
        'beamwidth': [65] * 6,
    }
    nr_hw_df = pd.DataFrame(nr_cr_data)
    nr_hw_nb, _, _ = find_neighbors(nr_hw_df, 5.0, True)
    nr_hw_sg, nr_hw_c2s = detect_sector_groups(nr_hw_df)
    nr_hw_plan = plan_rsi_network(nr_hw_df, nr_hw_nb, 'NR', nr_hw_sg, nr_hw_c2s)
    assert len(nr_hw_plan) == 6, f"NR Huawei RSI plan: {len(nr_hw_plan)} rows"
    nr_roots = dict(zip(nr_hw_plan['cell_id'], nr_hw_plan['roots_needed']))
    assert nr_roots['NR_HW_1_1'] > nr_roots['NR_HW_2_1'], \
        f"NR 14500m should need more roots ({nr_roots['NR_HW_1_1']}) than 3000m ({nr_roots['NR_HW_2_1']})"
    print(f"✅ NR Huawei cellRange RSI plan: 6 cells planned, roots verified")

    # ------------------------------------------------------------------
    # Cross-prefix site grouping (Huawei multi-band same physical site)
    # ------------------------------------------------------------------
    # BU1701 site: 4 band prefixes (E,L,T,Z) with different site_ids
    # but the same physical site.  Cells at the same sector (O-1/O-2/O-3)
    # must share the same PCI across all band prefixes.
    cross_data = {
        'cell_id': [
            'EBU1701D', 'EBU1701G',                     # E-band: O-2, O-3
            'LBU1701A', 'LBU1701D', 'LBU1701G',         # L-band: O-1, O-2, O-3
            'TBU1701A', 'TBU1701B', 'TBU1701D',         # T-band: O-1, O-1, O-2
            'TBU1701E', 'TBU1701G', 'TBU1701H',         #         O-2, O-3, O-3
            'ZBU1701A', 'ZBU1701D', 'ZBU1701G',         # Z-band: O-1, O-2, O-3
        ],
        # Different site_ids per band prefix (Huawei eNodeB naming)
        'site_id': [
            'EBU1701', 'EBU1701',
            'LBU1701', 'LBU1701', 'LBU1701',
            'TBU1701', 'TBU1701', 'TBU1701',
            'TBU1701', 'TBU1701', 'TBU1701',
            'ZBU1701', 'ZBU1701', 'ZBU1701',
        ],
        'sector': [
            'O-2', 'O-3',
            'O-1', 'O-2', 'O-3',
            'O-1', 'O-1', 'O-2',
            'O-2', 'O-3', 'O-3',
            'O-1', 'O-2', 'O-3',
        ],
        'latitude':  [37.72] * 14,
        'longitude': [30.29] * 14,
        'azimuth': [
            160, 280,                     # E-band
            40, 160, 280,                 # L-band
            40, 40, 160, 160, 280, 280,   # T-band
            40, 160, 280,                 # Z-band
        ],
        'pci': [
            160, 452,
            221, 160, 452,
            221, 221, 160, 160, 452, 452,
            221, 160, 452,
        ],
        'beamwidth': [65] * 14,
    }
    cross_df = pd.DataFrame(cross_data)

    # Test 1: detect_sector_groups merges cross-prefix sites
    cross_sg, cross_c2s = detect_sector_groups(cross_df)

    # All cells at sector O-1 must share the same cell_to_sector key
    o1_keys = set()
    for cid in ['LBU1701A', 'TBU1701A', 'TBU1701B', 'ZBU1701A']:
        k = cross_c2s.get(cid)
        assert k is not None, f"{cid} missing from cell_to_sector"
        o1_keys.add(k)
    assert len(o1_keys) == 1, f"O-1 cells should share ONE sector key, got {o1_keys}"

    o2_keys = set()
    for cid in ['EBU1701D', 'LBU1701D', 'TBU1701D', 'TBU1701E', 'ZBU1701D']:
        o2_keys.add(cross_c2s.get(cid))
    assert len(o2_keys) == 1, f"O-2 cells should share ONE sector key, got {o2_keys}"

    o3_keys = set()
    for cid in ['EBU1701G', 'LBU1701G', 'TBU1701G', 'TBU1701H', 'ZBU1701G']:
        o3_keys.add(cross_c2s.get(cid))
    assert len(o3_keys) == 1, f"O-3 cells should share ONE sector key, got {o3_keys}"

    # Sector keys for O-1, O-2, O-3 must be DIFFERENT (different physical sectors)
    all_sec_keys = o1_keys | o2_keys | o3_keys
    assert len(all_sec_keys) == 3, f"Expected 3 distinct sector keys, got {all_sec_keys}"
    print("✅ Cross-prefix site grouping: O-1/O-2/O-3 sectors merged correctly")

    # Test 2: T-band cells should NOT be marked as indoor (they're outdoor)
    _DETECTED_INDOOR_CELLS.clear()
    detect_sector_groups(cross_df)
    for cid in ['TBU1701A', 'TBU1701D', 'TBU1701G']:
        assert cid not in _DETECTED_INDOOR_CELLS, \
            f"{cid} wrongly marked as indoor (outdoor T-band at multi-band site)"
    print("✅ Cross-prefix: T-band cells NOT misclassified as indoor")

    # Test 3: PCI plan should give same PCI to all cells at same sector
    cross_nb, _, _ = find_neighbors(cross_df, 5.0, True)
    cross_sg2, cross_c2s2 = detect_sector_groups(cross_df)
    pci_plan = plan_pci_network(
        cross_df, cross_nb,
        sector_groups=cross_sg2, cell_to_sector=cross_c2s2,
        sa_iterations_override=5000, check_mod3=True, check_mod6=True, check_mod30=True
    )
    planned = {k: int(v) for k, v in zip(pci_plan['cell_id'], pci_plan['planned_pci'])}

    # All O-1 cells must have the same planned PCI
    o1_pcis = set(planned[c] for c in ['LBU1701A', 'TBU1701A', 'TBU1701B', 'ZBU1701A'])
    assert len(o1_pcis) == 1, f"O-1 cells should share PCI, got {o1_pcis}"

    # All O-2 cells must have the same planned PCI
    o2_pcis = set(planned[c] for c in ['EBU1701D', 'LBU1701D', 'TBU1701D', 'TBU1701E', 'ZBU1701D'])
    assert len(o2_pcis) == 1, f"O-2 cells should share PCI, got {o2_pcis}"

    # All O-3 cells must have the same planned PCI
    o3_pcis = set(planned[c] for c in ['EBU1701G', 'LBU1701G', 'TBU1701G', 'TBU1701H', 'ZBU1701G'])
    assert len(o3_pcis) == 1, f"O-3 cells should share PCI, got {o3_pcis}"

    # Different sectors should have different PCIs (co-site collision)
    assert o1_pcis != o2_pcis, "O-1 and O-2 must have different PCIs"
    assert o2_pcis != o3_pcis, "O-2 and O-3 must have different PCIs"
    assert o1_pcis != o3_pcis, "O-1 and O-3 must have different PCIs"

    # Different sectors should have different mod3 values
    o1_m3 = list(o1_pcis)[0] % 3
    o2_m3 = list(o2_pcis)[0] % 3
    o3_m3 = list(o3_pcis)[0] % 3
    assert len({o1_m3, o2_m3, o3_m3}) == 3, \
        f"3 sectors should have 3 different mod3 values, got O-1:{o1_m3} O-2:{o2_m3} O-3:{o3_m3}"

    print(f"✅ Cross-prefix PCI plan: O-1→{list(o1_pcis)[0]} O-2→{list(o2_pcis)[0]} "
          f"O-3→{list(o3_pcis)[0]} — all sectors unique, mod3 distinct")

    print("\n🎉 All tests passed!")

if __name__ == '__main__':
    main()
