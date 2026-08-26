"""
3GPP PCI/RSI Planning Tool v2.0 – Streamlit Web UI
====================================================
• PCI Collision / Confusion / Mod3 / Mod4(NR) / Mod6(LTE) / Mod30(LTE) detection
• RSI conflict detection based on PRACH cell range (Ncs, preamble format)
• LTE (4G) and NR (5G) support per 3GPP TS 36.211 / 38.211
• Interactive Folium map with sector directions, PCI/RSI hover info
• Downloadable HTML map
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, tempfile, os
from collections import defaultdict

from pci_engine import (
    run_full_analysis, build_neighbor_table, find_neighbors,
    haversine_distance, decompose_pci, compute_cell_prach_info,
    get_ncs, get_lte_preamble_format, cell_range_from_ncs,
    cell_range_from_format, preambles_per_root, roots_needed,
    rsi_overlap, derive_zcz_from_cell_range,
    plan_rsi_network, plan_pci_network,
    find_optimal_pci_rsi_for_new_cells,
    suggest_pci, suggest_rsi, rescan_pci_rsi_for_cells,
    parse_band_info, enrich_band_columns, detect_sector_groups,
    compute_health_score,
    detect_collisions, detect_confusions,
    detect_mod3_conflicts, detect_mod4_conflicts, detect_mod6_conflicts, detect_mod30_conflicts,
    detect_rsi_collisions,
    _is_same_site_by_id, _is_co_sector_by_id,
    get_sector_label, get_cell_environment, enrich_df_with_sector_info,
    _build_loc_az_maps,
    LTE_PCI_COUNT, NR_PCI_COUNT, LTE_NCS_UNRESTRICTED, NZC_LONG,
    LTE_PREAMBLE_FORMATS, NR_PREAMBLE_FORMATS, NR_NCS_LONG, NR_NCS_SHORT,
)
from data_handler import (
    read_excel_file, generate_sample_excel, export_results_to_excel
)

# ============================================================
st.set_page_config(page_title="TürkTelekom PCI/RSI Planner", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.tt-banner{
  display:flex;align-items:center;justify-content:center;gap:18px;
  padding:1rem 2rem;margin:-1rem -1rem 0.8rem -1rem;
  background:linear-gradient(135deg,#0d1b2a 0%,#1b2d4f 50%,#162a47 100%);
  border-radius:0 0 16px 16px;
  box-shadow:0 4px 20px rgba(0,0,0,0.3);
}
.tt-banner img{height:56px;background:white;border-radius:10px;padding:5px;
  box-shadow:0 2px 8px rgba(0,0,0,0.2)}
.tt-banner .tt-text{display:flex;flex-direction:column;line-height:1.15}
.tt-banner .tt-title{font-size:1.8rem;font-weight:800;
  background:linear-gradient(90deg,#42a5f5,#1e88e5,#00bcd4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:0.5px}
.tt-banner .tt-sub{font-size:0.78rem;color:#90a4ae;letter-spacing:3px;
  text-transform:uppercase;font-weight:500;margin-top:2px}
.critical{color:#FF1744;font-weight:bold} .high{color:#FF9100;font-weight:bold}
.medium{color:#FFC400;font-weight:bold} .low{color:#00E676;font-weight:bold}
[data-testid="stImage"] img {background:white;border-radius:8px;padding:4px}
</style>""", unsafe_allow_html=True)

_STATE_KEYS = ('df','results','neighbor_table','prach_info','map_html','rsi_plan','pci_plan','external_neighbors','sector_groups','cell_to_sector','_plan_cache_key','_plan_cache_result','_plan_detail_cache','pci_suggestions','rsi_suggestions','rescan_results','new_cell_results','_sug_cache_key','_sug_cache_result','analysis_params')
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

@st.cache_data(show_spinner='📂 Excel işleniyor...')
def _process_cell_excel(file_bytes: bytes, file_name: str):
    """Read + enrich the uploaded cell Excel once per file content.

    Cached so sidebar/widget interactions don't re-parse the file on
    every Streamlit rerun (critical for large networks)."""
    df, msgs = read_excel_file(io.BytesIO(file_bytes))
    sg, c2s = None, None
    if df is not None:
        df = enrich_band_columns(df)
        if (('prach_config_index' in df.columns and 'zero_correlation_zone' in df.columns)
                or 'cell_range' in df.columns):
            df['cell_range_km'] = df.apply(
                lambda r: compute_cell_prach_info(r).get('cell_range_ncs_km', None), axis=1)
        sg, c2s = detect_sector_groups(df, azimuth_tolerance=10.0)
    return df, msgs, sg, c2s

@st.cache_data(show_spinner=False)
def _read_external_neighbors(file_bytes: bytes):
    return pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image('tt_logo.svg', width=120)
    st.markdown("## TürkTelekom PCI/RSI Planner")
    st.markdown("---")
    technology = st.selectbox("Teknoloji", ["LTE (4G)", "NR (5G)"])
    tech = "LTE" if "LTE" in technology else "NR"
    radius_km = st.number_input("Tarama Yarıçapı (km)", min_value=0.1, max_value=50.0,
                                value=3.0, step=0.1, format="%.1f",
                                help="Mesafe+anten komşuluğu için tarama yarıçapı")
    use_antenna = st.checkbox("Anten Yönünü Kullan", True)
    default_bw = st.slider("Varsayılan Hüzme Genişliği (°)", 30, 120, 65, 5)
    st.markdown("---")
    st.markdown("### 🔗 Komşuluk Kaynakları")
    include_intra_site = st.checkbox("Aynı Site Sektörleri = Komşu", True,
                                     help="Aynı site_id\'ye sahip hücreler otomatik olarak komşu sayılır")
    st.markdown("""<small>ℹ️ Komşuluk 3 kaynaktan bulunur:<br>
    • <b>Mesafe+Anten:</b> Yarıçap içi + hüzme açısı<br>
    • <b>Aynı-Site:</b> Aynı site sektörleri<br>
    • <b>Harici Liste:</b> Excel komşuluk tablosu</small>""", unsafe_allow_html=True)
    nbr_upload = st.file_uploader("📋 Harici Komşuluk Listesi (opsiyonel)", type=['xlsx','xls'],
                                   help="Sütunlar: cell_1, cell_2, attempts (opsiyonel)")
    if nbr_upload:
        try:
            ext_df = _read_external_neighbors(nbr_upload.getvalue())
            st.session_state.external_neighbors = ext_df
            _att_count = sum(1 for c in ext_df.columns if str(c).strip().lower() in
                            ('attempts','attempt','ho_attempts','ho_attempt','handover',
                             'handovers','ho_count','ho','att','deneme','girisim','sayi'))
            _att_msg = f" (HO attempt sütunu algılandı ✅)" if _att_count > 0 else ""
            st.success(f"✅ {len(ext_df)} komşuluk ilişkisi yüklendi{_att_msg}")
        except Exception as e:
            st.error(f"❌ Okuma hatası: {e}")
    st.markdown("---")
    st.markdown("### 🔍 Kontroller")
    check_mod3 = st.checkbox("Mod 3 (PSS)", True)
    if tech == 'NR':
        check_mod4 = st.checkbox("Mod 4 (SSB DMRS)", True)
        check_mod6 = False
        check_mod30 = False
        st.caption("ℹ️ NR modunda Mod6/Mod30 devre dışıdır (LTE'ye özel).")
    else:
        check_mod4 = False
        check_mod6 = st.checkbox("Mod 6 (RS)", True)
        check_mod30 = st.checkbox("Mod 30 (PCFICH)", True)
    check_rsi = st.checkbox("RSI (PRACH cell-range)", True)
    st.markdown("---")
    @st.cache_data(show_spinner=False)
    def _cached_sample(t):
        return generate_sample_excel(technology=t)
    sample = _cached_sample(tech)
    st.download_button(f"📄 Örnek Excel İndir ({tech})", sample,
                       f"ornek_hucre_verisi_{tech.lower()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("---")
    if st.button("🗑️ Yeni Oturum — Verileri Temizle", use_container_width=True,
                 help="Yüklü veriyi, analiz sonuçlarını ve planları temizler"):
        for _k in _STATE_KEYS:
            st.session_state[_k] = None
        st.rerun()
    st.caption("3GPP TS 36.211 / 38.211 • v2.0")

# ============================================================
def _stale_results_warning():
    """Warn if sidebar parameters changed after the last analysis run."""
    ap = st.session_state.analysis_params
    if not ap or st.session_state.results is None:
        return
    _now = {'tech': tech, 'radius_km': float(radius_km),
            'use_antenna': use_antenna, 'default_bw': float(default_bw),
            'mod3': check_mod3, 'mod4': check_mod4, 'mod6': check_mod6,
            'mod30': check_mod30, 'rsi': check_rsi,
            'intra_site': include_intra_site,
            'has_external': st.session_state.external_neighbors is not None}
    _labels = {'tech': 'Teknoloji', 'radius_km': 'Tarama Yarıçapı',
               'use_antenna': 'Anten Yönü', 'default_bw': 'Hüzme Genişliği',
               'mod3': 'Mod3', 'mod4': 'Mod4', 'mod6': 'Mod6', 'mod30': 'Mod30',
               'rsi': 'RSI Kontrolü', 'intra_site': 'Aynı-Site Komşuluğu',
               'has_external': 'Harici Komşuluk Listesi'}
    _diff = [_labels[k] for k in _now if ap.get(k) != _now[k]]
    if _diff:
        st.warning(f"⚠️ Ayarlar son analizden sonra değişti (**{', '.join(_diff)}**) — "
                   "gösterilen sonuçlar önceki parametrelerle hesaplandı. "
                   "**📤 Veri Yükleme** sekmesinden analizi yeniden çalıştırın.")

# ============================================================
import base64, pathlib as _pathlib
_logo_b64 = base64.b64encode(_pathlib.Path('tt_logo.svg').read_bytes()).decode()
st.markdown(
    f'<div class="tt-banner">'
    f'<img src="data:image/svg+xml;base64,{_logo_b64}" alt="TT">'
    f'<div class="tt-text">'
    f'<span class="tt-title">TürkTelekom PCI / RSI Planner</span>'
    f'<span class="tt-sub">3GPP Compliant Planning &amp; Analysis</span>'
    f'</div></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Veri Yükleme", "📊 Analiz Sonuçları", "🗺️ Harita", "📋 Detaylı Raporlar",
    "💡 Öneriler & Tarama", "🆕 Yeni Hücre Ekle", "ℹ️ Bilgi"
])

# ============================================================
# TAB 1 — DATA UPLOAD
# ============================================================
with tab1:
    st.markdown("### 📤 Excel Dosyası Yükleme")
    c1, c2 = st.columns([2,1])
    with c1:
        uploaded = st.file_uploader("Hücre verisi Excel dosyası", type=['xlsx','xls'])
        if uploaded:
            # Cached by file content — reruns don't re-parse the Excel
            df, msgs, sg, c2s = _process_cell_excel(uploaded.getvalue(), uploaded.name)
            for m in msgs:
                (st.error if m.startswith("❌") else st.warning if m.startswith("⚠️")
                 else st.success if m.startswith("✅") else st.info)(m)
            if df is not None:
                st.session_state.df = df
                st.session_state.sector_groups = sg
                st.session_state.cell_to_sector = c2s
                st.success(f"✅ {len(df)} hücre yüklendi!")
                if sg:
                    n_co_cells = sum(len(v) for v in sg.values())
                    _col_groups = sum(1 for k in sg if k.startswith('COL_'))
                    if _col_groups > 0:
                        st.info(f"📶 {len(sg)} sektör grubu tespit edildi ({n_co_cells} co-sector hücre) — {_col_groups} grup sector kolonu referansıyla")
                    else:
                        st.info(f"📶 {len(sg)} sektör grubu tespit edildi ({n_co_cells} co-sector hücre, aynı konum + aynı yön ±10°)")
    with c2:
        st.markdown("""#### 📋 Gerekli Format
| Sütun | Zorunlu | Açıklama |
|-------|---------|----------|
| `cell_id` | ✅ | Benzersiz hücre ID |
| `latitude` | ✅ | Enlem |
| `longitude` | ✅ | Boylam |
| `pci` | ✅ | Physical Cell ID |
| `azimuth` | ❌ | Anten yönü (°) |
| `sector` | ❌ | Sektör numarası (dolu ise NC yerine bu kullanılır) |
| `rsi` | ❌ | Root Sequence Index |
| `prach_config_index` | ❌ | PRACH Config (0-63) |
| `zero_correlation_zone` | ❌ | Ncs config (0-15) |
| `cell_range` | ❌ | Huawei cellRadius (metre) — varsa ZCZ yerine kullanılır |
| `high_speed` | ❌ | highSpeedFlag (0/1) — restricted Ncs tablosu kullanılır |
""")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### 📋 Veri Önizleme")
        # Add sector label and environment columns for display
        _preview_df = df.copy()
        _preview_df.insert(1, 'sektör', _preview_df['cell_id'].astype(str).map(get_sector_label))
        _preview_df.insert(2, 'ortam', _preview_df['cell_id'].astype(str).map(get_cell_environment))
        st.dataframe(_preview_df, use_container_width=True, height=350)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Hücre", len(df))
        c2.metric("Benzersiz PCI", df['pci'].dropna().nunique())
        if 'rsi' in df.columns: c3.metric("Benzersiz RSI", df['rsi'].dropna().nunique())
        if 'site_id' in df.columns: c4.metric("Site", df['site_id'].nunique())

        # Band / Sector summary
        if 'band_label' in df.columns:
            st.markdown("### 📶 Band / Sektör Bilgileri")
            bc1, bc2 = st.columns(2)
            with bc1:
                band_counts = df['band_label'].value_counts()
                st.markdown("**Band Dağılımı:**")
                for bl, cnt in band_counts.items():
                    st.markdown(f"- **{bl}**: {cnt} hücre")
            with bc2:
                sg = st.session_state.sector_groups
                if sg:
                    multi_band_sectors = {k: v for k, v in sg.items() if len(v) > 1}
                    st.markdown(f"**Sektör Grupları:** {len(sg)} toplam")
                    st.markdown(f"- Çoklu band sektör: **{len(multi_band_sectors)}** (aynı PCI kullanmalı)")
                    st.markdown(f"- Tekli hücre sektör: **{len(sg) - len(multi_band_sectors)}**")
                    with st.expander("🔍 Sektör Gruplarını Gör"):
                        for sk, members in sorted(multi_band_sectors.items()):
                            bands = [df[df['cell_id']==m]['band_label'].values[0] if len(df[df['cell_id']==m]) > 0 else '?' for m in members]
                            st.markdown(f"**{sk}**: {', '.join([f'{m} ({b})' for m, b in zip(members, bands)])}")

        # PRACH info table
        st.markdown("### 📡 PRACH / Cell Range Bilgileri")
        prach_rows = []
        for _, row in df.iterrows():
            info = compute_cell_prach_info(row, tech)
            info['cell_id'] = row['cell_id']
            info['rsi'] = row.get('rsi')
            info['prach_config_index'] = row.get('prach_config_index', 0)
            info['zero_correlation_zone'] = info.get('effective_zcz', row.get('zero_correlation_zone', 5))
            cr = row.get('cell_range')
            if cr is not None and not pd.isna(cr):
                info['cell_range_input_m'] = int(cr)
            prach_rows.append(info)
        prach_df = pd.DataFrame(prach_rows)
        cols_order = ['cell_id','cell_range_input_m','prach_config_index','zero_correlation_zone','preamble_format',
                      'ncs','cell_range_ncs_km','cell_range_format_km',
                      'preambles_per_root','roots_needed','rsi']
        prach_df = prach_df[[c for c in cols_order if c in prach_df.columns]]
        prach_df = enrich_df_with_sector_info(prach_df)
        st.dataframe(prach_df, use_container_width=True, height=300)
        st.session_state.prach_info = prach_df

        st.markdown("---")
        if st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True):
            _prog_bar = st.progress(0, text='📊 Analiz başlıyor...')
            _prog_status = st.empty()
            def _analysis_progress(pct, msg=''):
                _prog_bar.progress(min(pct, 100), text=f'📊 {msg}' if msg else f'📊 Analiz... %{pct}')
            try:
                ext_nb = st.session_state.external_neighbors
                results = run_full_analysis(df, radius_km, tech, use_antenna,
                                            float(default_bw), check_mod3, check_mod6,
                                            check_mod30, check_rsi,
                                            include_intra_site, ext_nb,
                                            check_mod4=check_mod4,
                                            cell_to_sector=st.session_state.cell_to_sector,
                                            progress_callback=_analysis_progress)
                _prog_bar.progress(95, text='📊 Komşuluk tablosu oluşturuluyor...')
                st.session_state.results = results
                # Invalidate plan caches (scores will be recomputed with new analysis)
                st.session_state._plan_cache_key = None
                st.session_state._plan_cache_result = None
                st.session_state._plan_detail_cache = None
                _na = results.get('neighbor_attempts', {})
                st.session_state.neighbor_table = build_neighbor_table(
                    df, results['neighbors'], results.get('neighbor_sources'),
                    nbr_attempts=_na)
                _prog_bar.progress(100, text='✅ Analiz tamamlandı!')
                # Snapshot the parameters used — tabs warn if sidebar changes later
                st.session_state.analysis_params = {
                    'tech': tech, 'radius_km': float(radius_km),
                    'use_antenna': use_antenna, 'default_bw': float(default_bw),
                    'mod3': check_mod3, 'mod4': check_mod4, 'mod6': check_mod6,
                    'mod30': check_mod30, 'rsi': check_rsi,
                    'intra_site': include_intra_site,
                    'has_external': st.session_state.external_neighbors is not None}
                # Show neighbor source breakdown
                _src = results['summary'].get('neighbor_sources', {})
                _parts = []
                for k, v in sorted(_src.items(), key=lambda x: -x[1]):
                    _parts.append(f"**{k}**: {v}")
                if _parts:
                    st.info(f"🔗 Komşuluk kaynakları — {', '.join(_parts)}  \n"
                            f"ℹ️ Tarama yarıçapı ({radius_km} km) yalnızca *mesafe+anten* kaynaklı komşulukları etkiler. "
                            f"Harici komşuluk listesi ve aynı-site/aynı-konum komşulukları yarıçaptan bağımsızdır.")
                st.success("✅ Analiz tamamlandı — sonuçlar **📊 Analiz Sonuçları** sekmesinde.")
            except Exception as e:
                _prog_bar.empty()
                st.error(f"❌ Analiz hatası: {e}")

# ============================================================
# TAB 2 — RESULTS
# ============================================================
with tab2:
    if st.session_state.results is None:
        st.info("📌 Önce veri yükleyip analizi başlatın.")
    else:
        _stale_results_warning()
        import plotly.express as px
        import plotly.graph_objects as go
        results = st.session_state.results
        s = results['summary']

        # Health gauge — 3-way comparison: Mevcut vs Öneri vs Plan
        score = s['health_score']

        # Compute suggestion/plan scores if data available
        df = st.session_state.df
        nb = results['neighbors']
        c2s = st.session_state.cell_to_sector or {}
        tn = s['total_neighbor_pairs']

        def _recompute_score(pci_overrides, rsi_overrides, label):
            """Re-detect conflicts with overridden PCI/RSI values and compute score.
            pci_overrides / rsi_overrides: dict  cell_id(str) → int value
            Returns (score, col, con, m3, m4, m6, m30, rsi,
                     cosite_col, cosite_m3, cosite_m4, df_tmp)
            """
            df_tmp = df.copy()
            _cid_to_idx = {}
            for idx, cid_val in df_tmp['cell_id'].items():
                _cid_to_idx[str(cid_val)] = idx

            for cid, pval in pci_overrides.items():
                idx = _cid_to_idx.get(str(cid))
                if idx is not None:
                    df_tmp.at[idx, 'pci'] = pval
            if 'rsi' in df_tmp.columns:
                for cid, rval in rsi_overrides.items():
                    idx = _cid_to_idx.get(str(cid))
                    if idx is not None:
                        df_tmp.at[idx, 'rsi'] = rval

            _na = results.get('neighbor_attempts', {})
            _lac = _build_loc_az_maps(df_tmp)
            _col = detect_collisions(df_tmp, nb, cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac)
            _con = detect_confusions(df_tmp, nb, cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac)
            _m3 = detect_mod3_conflicts(df_tmp, nb, cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac) if check_mod3 else pd.DataFrame()
            _m4 = detect_mod4_conflicts(df_tmp, nb, cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac) if check_mod4 else pd.DataFrame()
            _m6 = detect_mod6_conflicts(df_tmp, nb, cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac) if check_mod6 else pd.DataFrame()
            _m30 = detect_mod30_conflicts(df_tmp, nb, cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac) if check_mod30 else pd.DataFrame()
            _rsi = detect_rsi_collisions(df_tmp, nb, 'rsi', s['technology'], cell_to_sector=c2s, nbr_attempts=_na, _loc_az_cache=_lac) if (check_rsi and 'rsi' in df_tmp.columns) else pd.DataFrame()
            hs = compute_health_score(len(_col), len(_con), len(_m3), len(_m6), len(_m30), len(_rsi), tn,
                                      n_cells=len(df_tmp), m4_count=len(_m4), technology=s['technology'])
            # Co-site counting
            _sm = dict(zip(df_tmp['cell_id'].astype(str), df_tmp['site_id'].astype(str))) if 'site_id' in df_tmp.columns else {}
            def _cs_cnt(cdf):
                if len(cdf) == 0: return 0
                _c2s = st.session_state.cell_to_sector or {}
                n = 0
                for _, rr in cdf.iterrows():
                    a, b = str(rr['cell_1']), str(rr['cell_2'])
                    # Skip co-sector pairs (they share PCI by design)
                    if _is_co_sector_by_id(a, b):
                        continue
                    if _c2s:
                        sa, sb = _c2s.get(a), _c2s.get(b)
                        if sa is not None and sa == sb:
                            continue
                    same = False
                    if _sm:
                        same = (_sm.get(a) == _sm.get(b) and _sm.get(a) not in (None,'','nan'))
                    if not same:
                        same = _is_same_site_by_id(a, b)
                    if same:
                        n += 1
                return n
            return (hs, len(_col), len(_con), len(_m3), len(_m4), len(_m6), len(_m30), len(_rsi),
                    _cs_cnt(_col), _cs_cnt(_m3), _cs_cnt(_m4), df_tmp)

        # Build override maps for plan scores
        plan_result = None
        pci_plan = st.session_state.pci_plan
        rsi_plan = st.session_state.rsi_plan
        if (pci_plan is not None and len(pci_plan) > 0) or (rsi_plan is not None and len(rsi_plan) > 0):
            pci_over = {}
            rsi_over = {}
            if pci_plan is not None and len(pci_plan) > 0:
                for _, r in pci_plan.iterrows():
                    sv = r.get('planned_pci', '—')
                    if str(sv) not in ('—','nan','None'):
                        pci_over[str(r['cell_id'])] = int(float(sv))
            if rsi_plan is not None and len(rsi_plan) > 0:
                for _, r in rsi_plan.iterrows():
                    sv = r.get('planned_rsi', '—')
                    if str(sv) not in ('—','nan','None'):
                        rsi_over[str(r['cell_id'])] = int(float(sv))
            if pci_over or rsi_over:
                # Cache key: frozenset of overrides to avoid recomputation on every rerun
                _cache_key = (frozenset(pci_over.items()), frozenset(rsi_over.items()))
                if st.session_state._plan_cache_key == _cache_key and st.session_state._plan_cache_result is not None:
                    plan_result = st.session_state._plan_cache_result
                else:
                    plan_result = _recompute_score(pci_over, rsi_over, "Plan")
                    st.session_state._plan_cache_key = _cache_key
                    st.session_state._plan_cache_result = plan_result

        # ── Suggestion score computation ──
        # Compute score even when suggestions are empty (0 rows = no issues).
        # This lets users see the comparison as soon as they click "Hesapla".
        sug_result = None
        _pci_sug_df = st.session_state.get('pci_suggestions')
        _rsi_sug_df = st.session_state.get('rsi_suggestions')
        if _pci_sug_df is not None or _rsi_sug_df is not None:
            _sug_pci_ov = {}
            _sug_rsi_ov = {}
            if _pci_sug_df is not None and len(_pci_sug_df) > 0:
                for _, _sr in _pci_sug_df.iterrows():
                    _sv = _sr.get('suggested_pci', '—')
                    if str(_sv) not in ('—','nan','None'):
                        _sug_pci_ov[str(_sr['cell_id'])] = int(float(_sv))
            if _rsi_sug_df is not None and len(_rsi_sug_df) > 0:
                for _, _sr in _rsi_sug_df.iterrows():
                    _sv = _sr.get('suggested_rsi', '—')
                    if str(_sv) not in ('—','nan','None'):
                        _sug_rsi_ov[str(_sr['cell_id'])] = int(float(_sv))
            _sug_ck = (frozenset(_sug_pci_ov.items()), frozenset(_sug_rsi_ov.items()))
            if st.session_state._sug_cache_key == _sug_ck and st.session_state._sug_cache_result is not None:
                sug_result = st.session_state._sug_cache_result
            else:
                if _sug_pci_ov or _sug_rsi_ov:
                    sug_result = _recompute_score(_sug_pci_ov, _sug_rsi_ov, 'Öneri')
                else:
                    # No actual changes → score stays same as current
                    sug_result = (score,
                                  s['collision_count'], s['confusion_count'],
                                  s['mod3_conflict_count'], s.get('mod4_conflict_count', 0),
                                  s['mod6_conflict_count'], s['mod30_conflict_count'],
                                  s['rsi_collision_count'],
                                  s.get('cosite_collision_count', 0),
                                  s.get('cosite_mod3_count', 0),
                                  s.get('cosite_mod4_count', 0), df)
                st.session_state._sug_cache_key = _sug_ck
                st.session_state._sug_cache_result = sug_result

        # Display gauges side by side
        n_gauges = 1 + (1 if sug_result else 0) + (1 if plan_result else 0)
        gauge_cols = st.columns(max(n_gauges, 2))
        def _make_gauge(val, title):
            gc = "#00E676" if val>=80 else "#FFC400" if val>=50 else "#FF1744"
            fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
                title={'text': title},
                number={'font': {'size': 48}},
                gauge={'axis':{'range':[0,100]},'bar':{'color':gc},
                       'steps':[{'range':[0,33],'color':'#FFCDD2'},
                                {'range':[33,66],'color':'#FFF9C4'},
                                {'range':[66,100],'color':'#C8E6C9'}]}))
            fig.update_layout(height=260, margin=dict(t=60, b=20, l=30, r=30))
            return fig
        _gi = 0
        with gauge_cols[_gi]:
            st.plotly_chart(_make_gauge(score, "🔍 Mevcut Skor"), use_container_width=True)
        if sug_result is not None:
            _gi += 1
            sug_score_val = sug_result[0]
            sug_delta = round(sug_score_val - score, 2)
            with gauge_cols[_gi]:
                st.plotly_chart(_make_gauge(sug_score_val, f"💡 Öneri Sonrası ({'+' if sug_delta>=0 else ''}{sug_delta})"), use_container_width=True)
        if plan_result is not None:
            _gi += 1
            plan_score_val = plan_result[0]
            delta = round(plan_score_val - score, 2)
            with gauge_cols[_gi]:
                st.plotly_chart(_make_gauge(plan_score_val, f"🚀 Plan Sonrası ({'+' if delta>=0 else ''}{delta})"), use_container_width=True)

        # Comparison table (co-sector conflicts are already excluded at detection time)
        _cs_col_cur = s.get('cosite_collision_count', 0)
        if sug_result is not None or plan_result is not None:
            _labels = ['Collision','  ↳ co-site','Confusion','Mod3','  ↳ co-site','Mod4','Mod6','Mod30','RSI','Skor']
            _mev = [s['collision_count'], _cs_col_cur,
                    s['confusion_count'],
                    s['mod3_conflict_count'], s.get('cosite_mod3_count',0),
                    s.get('mod4_conflict_count',0),
                    s['mod6_conflict_count'],s['mod30_conflict_count'],
                    s['rsi_collision_count'], score]
            comp_data = {'Tür': _labels, 'Mevcut': _mev}
            if sug_result is not None:
                _sg = sug_result
                comp_data['Öneri Sonrası'] = [
                    _sg[1], _sg[8], _sg[2], _sg[3], _sg[9],
                    _sg[4], _sg[5], _sg[6], _sg[7], _sg[0]]
            if plan_result is not None:
                _p = plan_result
                comp_data['Plan Sonrası'] = [
                    _p[1], _p[8], _p[2], _p[3], _p[9],
                    _p[4], _p[5], _p[6], _p[7], _p[0]]
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
            st.caption("ℹ️ Co-sektör (aynı site + aynı azimut) hücreleri arasındaki çakışmalar doğal olduğu için otomatik olarak hariç tutulmuştur.")

        _tech_now = s['technology']
        _cs_col = s.get('cosite_collision_count', 0)
        _cs_m3 = s.get('cosite_mod3_count', 0)
        _cs_m4 = s.get('cosite_mod4_count', 0)
        if _tech_now == 'NR':
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("🔴 Collision", s['collision_count'], help=f"Co-site: {_cs_col}")
            m2.metric("🟠 Confusion", s['confusion_count'])
            m3.metric("🟡 Mod3", s['mod3_conflict_count'], help=f"Co-site: {_cs_m3}")
            m4.metric("🟤 Mod4 (SSB)", s.get('mod4_conflict_count', 0), help=f"Co-site: {_cs_m4}")
            m5.metric("🟣 RSI", s['rsi_collision_count'])
        else:
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("🔴 Collision", s['collision_count'], help=f"Co-site: {_cs_col}")
            m2.metric("🟠 Confusion", s['confusion_count'])
            m3.metric("🟡 Mod3", s['mod3_conflict_count'], help=f"Co-site: {_cs_m3}")
            m4.metric("🔵 Mod6", s['mod6_conflict_count'])
            m5.metric("⚪ Mod30", s['mod30_conflict_count'])
            m6.metric("🟣 RSI", s['rsi_collision_count'])
        # Co-site conflict warning
        if _cs_col > 0 or _cs_m3 > 0 or _cs_m4 > 0:
            _warn_parts = []
            if _cs_col > 0:
                _warn_parts.append(f"**{_cs_col}** co-site Collision (aynı site, farklı sektör — aynı PCI)")
            if _cs_m3 > 0:
                _warn_parts.append(f"**{_cs_m3}** co-site Mod3 (aynı site, farklı sektör — PSS aynı)")
            if _cs_m4 > 0:
                _warn_parts.append(f"**{_cs_m4}** co-site Mod4 (aynı site, farklı sektör — SSB DMRS aynı)")
            st.warning("⚠️ " + ", ".join(_warn_parts) + ". Co-site hücrelerinde PCI/mod3/mod4 **kesinlikle** farklı olmalıdır!")

        st.markdown("---")
        cl, cr = st.columns(2)
        with cl:
            st.markdown("### 📋 Genel")
            nb_src = s.get('neighbor_sources', {})
            src_text = ', '.join([f"{k}: {v}" for k,v in nb_src.items()]) if nb_src else '—'
            st.table(pd.DataFrame({
                'Metrik':['Teknoloji','Hücre','Komşu Çifti','Yarıçap','PCI Aralığı',
                          'Sorunlu Hücre','Komşuluk Kaynakları'],
                'Değer':[str(s['technology']),str(s['total_cells']),str(s['total_neighbor_pairs']),
                         f"{s['search_radius_km']} km",str(s['max_pci_range']),str(s['cells_with_issues']),
                         src_text]}))
        with cr:
            st.markdown("### 📊 Sorun Dağılımı")
            if s['technology'] == 'NR':
                idf = pd.DataFrame({
                    'Tip':[f'Collision (cs:{_cs_col})','Confusion',
                           f'Mod3 (cs:{_cs_m3})',f'Mod4 (cs:{_cs_m4})','RSI'],
                    'Sayı':[s['collision_count'],s['confusion_count'],s['mod3_conflict_count'],
                            s.get('mod4_conflict_count',0),s['rsi_collision_count']],
                    'Önem':['KRİTİK','YÜKSEK','ORTA','ORTA','YÜKSEK']})
            else:
                idf = pd.DataFrame({
                    'Tip':[f'Collision (cs:{_cs_col})','Confusion',
                           f'Mod3 (cs:{_cs_m3})','Mod6','Mod30','RSI'],
                    'Sayı':[s['collision_count'],s['confusion_count'],s['mod3_conflict_count'],
                            s['mod6_conflict_count'],s['mod30_conflict_count'],s['rsi_collision_count']],
                    'Önem':['KRİTİK','YÜKSEK','ORTA','DÜŞÜK','DÜŞÜK','YÜKSEK']})
            fb = px.bar(idf, x='Tip', y='Sayı', color='Önem', text='Sayı',
                        color_discrete_map={'KRİTİK':'#FF1744','YÜKSEK':'#FF9100',
                                            'ORTA':'#FFC400','DÜŞÜK':'#00E676'})
            fb.update_layout(height=320); st.plotly_chart(fb, use_container_width=True)

        # PCI dist
        df = st.session_state.df
        ca, cb = st.columns(2)
        with ca:
            fh = px.histogram(df, x='pci', nbins=50, title='PCI Histogram')
            fh.update_layout(height=320); st.plotly_chart(fh, use_container_width=True)
        with cb:
            dmod = df.copy(); dmod['mod3'] = dmod['pci'].apply(lambda x: int(x)%3 if pd.notna(x) else None)
            fp = px.pie(dmod.dropna(subset=['mod3']), names='mod3', title='PCI Mod 3 (PSS)',
                        color_discrete_sequence=['#FF6384','#36A2EB','#FFCE56'])
            fp.update_layout(height=320); st.plotly_chart(fp, use_container_width=True)

        st.markdown("---")
        # Lazy Excel generation — only build on button click to avoid MemoryError
        if 'analysis_excel_bytes' not in st.session_state:
            st.session_state['analysis_excel_bytes'] = None
        _xl_col1, _xl_col2 = st.columns([3, 1])
        with _xl_col1:
            if st.button("📥 Excel Oluştur ve İndir", type="primary", use_container_width=True):
                with st.spinner("Excel dosyası oluşturuluyor..."):
                    try:
                        nt = st.session_state.neighbor_table
                        pi = st.session_state.prach_info
                        xl = export_results_to_excel(df, results, nt, pi)
                        st.session_state['analysis_excel_bytes'] = xl
                    except Exception as _xle:
                        st.error(f"❌ Excel oluşturma hatası: {_xle}")
        with _xl_col2:
            if st.session_state.get('analysis_excel_bytes') is not None:
                st.download_button("⬇️ İndir", st.session_state['analysis_excel_bytes'],
                                   "pci_rsi_analiz_sonuclari.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

        # ============================================================
        # FULL NETWORK AUTO-PLANNER (inside TAB 2)
        # ============================================================
        st.markdown("---")
        st.markdown("## 🚀 Otomatik PCI / RSI Planlama")
        st.markdown("""> Analiz sonuçlarını inceledikten sonra **tüm ağ için** optimal PCI ve RSI planlaması yapabilirsiniz.
> PCI için **Simulated Annealing (SA)** metaheuristic optimizasyon kullanılır.
> RSI için cell-range-aware greedy algoritma kullanılır.""")

        # --- SA Parameters Expander ---
        with st.expander("⚙️ SA Planlama Parametreleri", expanded=False):
            st.markdown("SA optimizasyon parametrelerini özelleştirin. Varsayılan değerler çoğu ağ için uygundur.")
            _sa_c1, _sa_c2 = st.columns(2)
            with _sa_c1:
                sa_iter_input = st.number_input(
                    "🔄 SA İterasyon Sayısı",
                    min_value=0, max_value=5_000_000, value=0, step=100_000,
                    help="0 = Otomatik (ağ büyüklüğüne göre 300K-800K). Daha fazla iterasyon daha iyi sonuç verebilir ancak daha uzun sürer.")
                st.caption("0 = otomatik hesaplama")
            with _sa_c2:
                sa_reserved_enabled = st.checkbox("📌 Rezerv PCI Aralığı", value=False,
                    help="Belirli PCI değerlerini planlama dışı bırakın (örn: özel kullanım için ayrılmış PCI'lar)")
            if sa_reserved_enabled:
                _res_c1, _res_c2 = st.columns(2)
                _max_pci_ui = 1007 if results['summary']['technology'] == 'NR' else 503
                with _res_c1:
                    sa_reserved_start = st.number_input("Rezerv Başlangıç PCI", min_value=0, max_value=_max_pci_ui, value=0)
                with _res_c2:
                    sa_reserved_end = st.number_input("Rezerv Bitiş PCI", min_value=0, max_value=_max_pci_ui, value=0)
                if sa_reserved_start <= sa_reserved_end:
                    _n_reserved = sa_reserved_end - sa_reserved_start + 1
                    st.info(f"📌 PCI **{sa_reserved_start}** — **{sa_reserved_end}** arası (**{_n_reserved}** değer) planlama dışı bırakılacak.")
                else:
                    st.warning("⚠️ Başlangıç değeri bitiş değerinden büyük olamaz.")
                    sa_reserved_start = None
                    sa_reserved_end = None
            else:
                sa_reserved_start = None
                sa_reserved_end = None

        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("📡 Tüm Ağ İçin PCI Planla (SA)", type="primary", use_container_width=True):
                try:
                    _pci_prog = st.progress(0, text='🧠 SA başlatılıyor...')
                    def _pci_progress(pct, msg=''):
                        _pci_prog.progress(min(pct, 100), text=f'🧠 {msg}' if msg else f'🧠 PCI planlama... %{pct}')
                    _df = st.session_state.df
                    _results = st.session_state.results
                    _tech = _results['summary']['technology']
                    pci_plan = plan_pci_network(_df, _results['neighbors'], _tech,
                                               check_mod3, check_mod6, check_mod30,
                                               st.session_state.sector_groups,
                                               st.session_state.cell_to_sector,
                                               check_mod4=check_mod4,
                                               nbr_attempts=_results.get('neighbor_attempts', {}),
                                               progress_callback=_pci_progress,
                                               sa_iterations_override=sa_iter_input,
                                               reserved_pci_start=sa_reserved_start,
                                               reserved_pci_end=sa_reserved_end)
                    st.session_state.pci_plan = pci_plan
                    # Invalidate plan caches
                    st.session_state._plan_cache_key = None
                    st.session_state._plan_cache_result = None
                    st.session_state._plan_detail_cache = None
                    _pci_prog.progress(100, text=f'✅ PCI planı tamamlandı — {len(pci_plan)} hücre')
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ PCI planlama hatası: {e}")
        with pc2:
            if st.button("📻 Tüm Ağ İçin RSI Planla", type="primary", use_container_width=True):
                try:
                    _rsi_prog = st.progress(0, text='📻 RSI planlama başlıyor...')
                    def _rsi_progress(pct, msg=''):
                        _rsi_prog.progress(min(pct, 100), text=f'📻 {msg}' if msg else f'📻 RSI planlama... %{pct}')
                    _df = st.session_state.df
                    _results = st.session_state.results
                    _tech = _results['summary']['technology']
                    rsi_plan = plan_rsi_network(_df, _results['neighbors'], _tech,
                                                   st.session_state.sector_groups,
                                                   st.session_state.cell_to_sector,
                                                   progress_callback=_rsi_progress)
                    st.session_state.rsi_plan = rsi_plan
                    # Invalidate plan caches
                    st.session_state._plan_cache_key = None
                    st.session_state._plan_cache_result = None
                    st.session_state._plan_detail_cache = None
                    _rsi_prog.progress(100, text=f'✅ RSI planı tamamlandı — {len(rsi_plan)} hücre')
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ RSI planlama hatası: {e}")

        # --- Show PCI Plan ---
        pci_plan = st.session_state.pci_plan
        if pci_plan is not None and len(pci_plan) > 0:
            st.markdown("### 📡 PCI Otomatik Plan Sonuçları")

            total = len(pci_plan)
            assigned_ok = len(pci_plan[pci_plan['planned_pci'] != '—'])
            unassigned = total - assigned_ok
            same = len(pci_plan[pci_plan['changed'] == '— Aynı'])
            changed = assigned_ok - same

            st.markdown(f"""- **{total}** hücre için PCI planlandı
- ✅ Atanan: **{assigned_ok}** ({100*assigned_ok//total}%)
- 🔄 Değişen: **{changed}**
- — Aynı kalan: **{same}**
- ❌ Atanamayan: **{unassigned}**""")

            if 'relaxation_level' in pci_plan.columns:
                level_counts = pci_plan['relaxation_level'].value_counts()
                st.markdown("#### 📊 Atama Durumu Dağılımı")
                level_df = pd.DataFrame({
                    'Seviye': level_counts.index,
                    'Hücre Sayısı': level_counts.values
                })
                st.dataframe(level_df, use_container_width=True, hide_index=True)

            st.dataframe(pci_plan, use_container_width=True, height=450)

            # ── Post-plan analysis charts ──────────────────────────
            if plan_result is not None:
                st.markdown("#### 📊 Plan Sonrası Analiz")
                _p = plan_result
                # _p = (hs, col, con, m3, m4, m6, m30, rsi, cs_col, cs_m3, cs_m4, df_tmp)
                _plan_df = _p[11]  # df with planned PCI values applied

                # Sorun Dağılımı bar chart
                p_cl, p_cr = st.columns(2)
                with p_cl:
                    if _tech_now == 'NR':
                        p_idf = pd.DataFrame({
                            'Tip':[f'Collision (cs:{_p[8]})','Confusion',
                                   f'Mod3 (cs:{_p[9]})',f'Mod4 (cs:{_p[10]})','RSI'],
                            'Sayı':[_p[1],_p[2],_p[3],_p[4],_p[7]],
                            'Önem':['KRİTİK','YÜKSEK','ORTA','ORTA','YÜKSEK']})
                    else:
                        p_idf = pd.DataFrame({
                            'Tip':[f'Collision (cs:{_p[8]})','Confusion',
                                   f'Mod3 (cs:{_p[9]})','Mod6','Mod30','RSI'],
                            'Sayı':[_p[1],_p[2],_p[3],_p[5],_p[6],_p[7]],
                            'Önem':['KRİTİK','YÜKSEK','ORTA','DÜŞÜK','DÜŞÜK','YÜKSEK']})
                    p_fb = px.bar(p_idf, x='Tip', y='Sayı', color='Önem', text='Sayı',
                                 title='Plan Sonrası Sorun Dağılımı',
                                 color_discrete_map={'KRİTİK':'#FF1744','YÜKSEK':'#FF9100',
                                                     'ORTA':'#FFC400','DÜŞÜK':'#00E676'})
                    p_fb.update_layout(height=320)
                    st.plotly_chart(p_fb, use_container_width=True)

                with p_cr:
                    # PSS (mod3) pie chart for planned state
                    _plan_mod3 = _plan_df.copy()
                    _plan_mod3['PSS (mod3)'] = _plan_mod3['pci'].apply(lambda x: int(x) % 3 if pd.notna(x) else None)
                    p_fp = px.pie(_plan_mod3.dropna(subset=['PSS (mod3)']), names='PSS (mod3)',
                                 title='Plan Sonrası PCI Mod3 (PSS) Dağılımı',
                                 color_discrete_sequence=['#FF6384','#36A2EB','#FFCE56'])
                    p_fp.update_layout(height=320)
                    st.plotly_chart(p_fp, use_container_width=True)

                # PCI histogram for planned state
                p_fh = px.histogram(_plan_df, x='pci', nbins=50, title='Plan Sonrası PCI Histogram')
                p_fh.update_layout(height=300)
                st.plotly_chart(p_fh, use_container_width=True)

        # --- Show RSI Plan ---
        rsi_plan = st.session_state.rsi_plan
        if rsi_plan is not None and len(rsi_plan) > 0:
            st.markdown("### 📻 RSI Otomatik Plan Sonuçları")
            st.markdown(f"""- **{len(rsi_plan)}** hücre için RSI planlandı
- ✅ Değişen: **{len(rsi_plan[rsi_plan['changed']=='✅ Değişti'])}**
- — Aynı kalan: **{len(rsi_plan[rsi_plan['changed']=='— Aynı'])}**
- ❌ Atanamayan: **{len(rsi_plan[rsi_plan['planned_rsi']=='—'])}**""")

            st.dataframe(rsi_plan, use_container_width=True, height=450)

        # Download all plans
        _pci_plan = st.session_state.pci_plan
        _rsi_plan = st.session_state.rsi_plan
        has_pci_plan = _pci_plan is not None and len(_pci_plan) > 0
        has_rsi_plan = _rsi_plan is not None and len(_rsi_plan) > 0
        if has_pci_plan or has_rsi_plan:
            plan_buf = io.BytesIO()
            with pd.ExcelWriter(plan_buf, engine='xlsxwriter') as w:
                if has_pci_plan:
                    _pci_plan.to_excel(w, sheet_name='PCI Planı', index=False)
                if has_rsi_plan:
                    _rsi_plan.to_excel(w, sheet_name='RSI Planı', index=False)
            st.download_button("📥 Otomatik Planı Excel Olarak İndir", plan_buf.getvalue(),
                               "pci_rsi_oto_plan.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

# ============================================================
# TAB 3 — MAP  (Lazy-loaded for performance)
# ============================================================
with tab3:
    if st.session_state.df is None:
        st.info("📌 Önce veri yükleyin.")
    else:
        import folium
        from folium.plugins import MarkerCluster
        df_orig = st.session_state.df
        results = st.session_state.results

        st.markdown("### 🗺️ PCI/RSI Haritası")
        _stale_results_warning()

        # --- Data source selector ---
        map_sources = ["📋 Mevcut (Orijinal) Değerler"]
        pci_plan = st.session_state.pci_plan
        rsi_plan = st.session_state.rsi_plan
        has_plan = ((pci_plan is not None and len(pci_plan) > 0) or
                    (rsi_plan is not None and len(rsi_plan) > 0))
        if has_plan:
            map_sources.append("🚀 Otomatik Plan Sonrası Değerler")

        data_source = st.selectbox("📊 Harita Veri Kaynağı", map_sources,
                                   help="Haritada gösterilecek PCI/RSI değerlerini seçin. Plan sonrası değerleri seçerek kıyaslama yapabilirsiniz.")

        mc1, mc2, mc3 = st.columns(3)
        show_neighbors = mc1.checkbox("Komşuluk Hatları", False)
        show_conflicts = mc2.checkbox("Çakışmaları Vurgula", True)
        base_sector_m = mc3.slider("Baz sektör uzunluğu (m)", 50, 2000, 350, 50)

        _all_conflict_types = ['Collision', 'Confusion', 'Mod3', 'Mod4', 'Mod6', 'Mod30', 'RSI']
        selected_conflict_types = _all_conflict_types if show_conflicts else []

        # --- Build effective PCI/RSI maps based on data source ---
        def _build_effective_maps(df_orig, data_source, pci_plan, rsi_plan):
            """Build effective PCI/RSI dicts and change-tracking sets from selected data source."""
            eff_pci = dict(zip(df_orig['cell_id'].astype(str), df_orig['pci']))
            eff_rsi = dict(zip(df_orig['cell_id'].astype(str), df_orig.get('rsi', pd.Series(dtype=float)))) if 'rsi' in df_orig.columns else {}
            changed_cells = set()  # cells whose PCI or RSI changed
            source_label = "Mevcut"

            if "Plan" in data_source:
                source_label = "Plan Sonrası"
                if pci_plan is not None and len(pci_plan) > 0:
                    for _, r in pci_plan.iterrows():
                        cid = str(r['cell_id'])
                        pl = r.get('planned_pci', '—')
                        if str(pl) != '—' and str(pl) != 'nan':
                            eff_pci[cid] = int(float(pl))
                            if str(r.get('changed', '')) == '✅ Değişti':
                                changed_cells.add(cid)
                if rsi_plan is not None and len(rsi_plan) > 0:
                    for _, r in rsi_plan.iterrows():
                        cid = str(r['cell_id'])
                        pl = r.get('planned_rsi', '—')
                        if str(pl) != '—' and str(pl) != 'nan':
                            eff_rsi[cid] = int(float(pl))
                            if str(r.get('changed', '')) == '✅ Değişti':
                                changed_cells.add(cid)

            return eff_pci, eff_rsi, changed_cells, source_label

        # Build map only when user clicks the button
        if st.button("🗺️ Haritayı Oluştur / Güncelle", type="primary", use_container_width=True):
            with st.spinner(f"Harita oluşturuluyor ({len(df_orig)} hücre)..."):
                eff_pci, eff_rsi, changed_cells, source_label = _build_effective_maps(
                    df_orig, data_source, pci_plan, rsi_plan)

                # Conflict cell set with per-cell type classification
                # For plan/suggestion views: re-detect conflicts with effective values
                conflict_cells = set()
                cell_conflict_types = defaultdict(set)  # cell_id → {'Collision','Confusion','Mod3',...}
                _conflict_tables_data = {}  # clabel → DataFrame (used later for legend counts)

                if results and show_conflicts:
                    nb = results['neighbors']
                    c2s = st.session_state.cell_to_sector or {}
                    _na = results.get('neighbor_attempts', {})
                    tech = results['summary']['technology']

                    if "Mevcut" not in data_source:
                        # Re-detect conflicts using effective PCI/RSI values
                        df_eff = df_orig.copy()
                        _cid_idx = {}
                        for idx, cid_val in df_eff['cell_id'].items():
                            _cid_idx[str(cid_val)] = idx
                        for cid, pval in eff_pci.items():
                            ix = _cid_idx.get(str(cid))
                            if ix is not None:
                                df_eff.at[ix, 'pci'] = pval
                        if 'rsi' in df_eff.columns:
                            for cid, rval in eff_rsi.items():
                                ix = _cid_idx.get(str(cid))
                                if ix is not None:
                                    df_eff.at[ix, 'rsi'] = rval
                        _conflict_tables_data['Collision'] = detect_collisions(df_eff, nb, cell_to_sector=c2s, nbr_attempts=_na)
                        _conflict_tables_data['Confusion'] = detect_confusions(df_eff, nb, cell_to_sector=c2s, nbr_attempts=_na)
                        _conflict_tables_data['Mod3'] = detect_mod3_conflicts(df_eff, nb, cell_to_sector=c2s, nbr_attempts=_na)
                        _conflict_tables_data['Mod4'] = detect_mod4_conflicts(df_eff, nb, cell_to_sector=c2s, nbr_attempts=_na)
                        _conflict_tables_data['Mod6'] = detect_mod6_conflicts(df_eff, nb, cell_to_sector=c2s, nbr_attempts=_na)
                        _conflict_tables_data['Mod30'] = detect_mod30_conflicts(df_eff, nb, cell_to_sector=c2s, nbr_attempts=_na)
                        _conflict_tables_data['RSI'] = detect_rsi_collisions(df_eff, nb, 'rsi', tech, cell_to_sector=c2s, nbr_attempts=_na) if 'rsi' in df_eff.columns else pd.DataFrame()
                    else:
                        # Mevcut: use existing results
                        _key_map = {'Collision':'collisions','Confusion':'confusions',
                                    'Mod3':'mod3_conflicts','Mod4':'mod4_conflicts',
                                    'Mod6':'mod6_conflicts',
                                    'Mod30':'mod30_conflicts','RSI':'rsi_collisions'}
                        for clabel, ckey in _key_map.items():
                            _conflict_tables_data[clabel] = results.get(ckey, pd.DataFrame())

                    for clabel, tbl in _conflict_tables_data.items():
                        if clabel not in selected_conflict_types:
                            continue
                        if len(tbl) > 0:
                            for _, crow in tbl.iterrows():
                                c1, c2 = str(crow['cell_1']), str(crow['cell_2'])
                                conflict_cells.add(c1)
                                conflict_cells.add(c2)
                                cell_conflict_types[c1].add(clabel)
                                cell_conflict_types[c2].add(clabel)

                # For suggestion/plan views: mark cells that changed & had conflicts as resolved
                resolved_cells = set()
                # (Cells now only show as conflicting if they STILL have a conflict after re-detection)

                center_lat = df_orig['latitude'].mean()
                center_lon = df_orig['longitude'].mean()
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                               tiles='OpenStreetMap')

                # --- Band visual configuration (highly distinct colors) ---
                band_config = {
                    '700 MHz':           {'hex': '#1B5E20', 'hex_fill': '#66BB6A', 'coverage_scale': 1.8,  'draw_order': 0,  'icon_color': 'darkgreen', 'dash': None},
                    '800 MHz':           {'hex': '#2E7D32', 'hex_fill': '#43A047', 'coverage_scale': 1.6,  'draw_order': 1,  'icon_color': 'green',     'dash': None},
                    '900 MHz':           {'hex': '#00695C', 'hex_fill': '#26A69A', 'coverage_scale': 1.5,  'draw_order': 2,  'icon_color': 'cadetblue', 'dash': None},
                    '1800 MHz':          {'hex': '#6A1B9A', 'hex_fill': '#AB47BC', 'coverage_scale': 1.0,  'draw_order': 3,  'icon_color': 'purple',    'dash': None},
                    '1800 MHz / 20 MHz': {'hex': '#1565C0', 'hex_fill': '#42A5F5', 'coverage_scale': 1.0,  'draw_order': 4,  'icon_color': 'blue',      'dash': None},
                    '1800 MHz / 10 MHz': {'hex': '#E65100', 'hex_fill': '#FF9800', 'coverage_scale': 0.9,  'draw_order': 5,  'icon_color': 'orange',    'dash': '5'},
                    '2100 MHz':          {'hex': '#C62828', 'hex_fill': '#EF5350', 'coverage_scale': 0.75, 'draw_order': 6,  'icon_color': 'red',       'dash': None},
                    '2300 MHz':          {'hex': '#AD1457', 'hex_fill': '#EC407A', 'coverage_scale': 0.65, 'draw_order': 7,  'icon_color': 'pink',      'dash': None},
                    '2600 MHz':          {'hex': '#0277BD', 'hex_fill': '#29B6F6', 'coverage_scale': 0.55, 'draw_order': 8,  'icon_color': 'lightblue', 'dash': None},
                    '3500 MHz':          {'hex': '#F57F17', 'hex_fill': '#FFEE58', 'coverage_scale': 0.4,  'draw_order': 9,  'icon_color': 'beige',     'dash': None},
                    '3700 MHz':          {'hex': '#FF6F00', 'hex_fill': '#FFB74D', 'coverage_scale': 0.35, 'draw_order': 10, 'icon_color': 'orange',    'dash': None},
                    'Bilinmeyen':        {'hex': '#616161', 'hex_fill': '#9E9E9E', 'coverage_scale': 0.8,  'draw_order': 99, 'icon_color': 'gray',      'dash': '3'},
                }
                # Dynamic color for bands not in config — assign unique vibrant colors
                _extra_colors = [
                    ('#4A148C', '#CE93D8'), ('#004D40', '#80CBC4'), ('#BF360C', '#FF8A65'),
                    ('#1A237E', '#7986CB'), ('#33691E', '#AED581'), ('#880E4F', '#F48FB1'),
                    ('#311B92', '#B39DDB'), ('#01579B', '#4FC3F7'), ('#E65100', '#FFB74D'),
                ]
                _extra_idx = 0
                for bl in (df_orig['band_label'].unique() if 'band_label' in df_orig.columns else []):
                    if bl not in band_config:
                        eh, ef = _extra_colors[_extra_idx % len(_extra_colors)]
                        band_config[bl] = {'hex': eh, 'hex_fill': ef, 'coverage_scale': 0.7,
                                           'draw_order': 50 + _extra_idx, 'icon_color': 'gray', 'dash': None}
                        _extra_idx += 1

                # Create FeatureGroups per band
                band_labels_present = sorted(df_orig['band_label'].unique()) if 'band_label' in df_orig.columns else []
                band_fg = {}
                for bl in sorted(band_labels_present, key=lambda x: band_config.get(x, band_config['Bilinmeyen'])['draw_order']):
                    fg = folium.FeatureGroup(name=f"📶 {bl}", show=True)
                    fg.add_to(m)
                    band_fg[bl] = fg

                site_fg = folium.FeatureGroup(name="🗼 Site İşaretçileri", show=True)
                site_fg.add_to(m)

                sec_map = st.session_state.cell_to_sector or {}
                site_markers_placed = set()

                # Original PCI/RSI for comparison display
                orig_pci = dict(zip(df_orig['cell_id'].astype(str), df_orig['pci']))
                orig_rsi = dict(zip(df_orig['cell_id'].astype(str), df_orig['rsi'])) if 'rsi' in df_orig.columns else {}

                # --- Draw sectors per band ---
                for bl in sorted(band_labels_present, key=lambda x: band_config.get(x, band_config['Bilinmeyen'])['draw_order']):
                    cfg = band_config.get(bl, band_config['Bilinmeyen'])
                    fg = band_fg[bl]
                    band_df = df_orig[df_orig['band_label'] == bl] if 'band_label' in df_orig.columns else df_orig

                    for _, row in band_df.iterrows():
                        cid = str(row['cell_id'])
                        lat, lon = row['latitude'], row['longitude']
                        az = row.get('azimuth', 0) or 0
                        bw = row.get('beamwidth', 65) or 65
                        band_mhz = row.get('band_mhz', '')
                        bw_mhz = row.get('bandwidth_mhz', '')
                        site_id = row.get('site_id', '')
                        sec_key = sec_map.get(cid, '')

                        # Use effective (possibly changed) PCI/RSI values
                        pci = eff_pci.get(cid)
                        rsi = eff_rsi.get(cid)
                        pss, sss = decompose_pci(int(pci)) if pci is not None and not pd.isna(pci) else ('?', '?')

                        is_conflict = cid in conflict_cells
                        is_resolved = cid in resolved_cells
                        is_changed = cid in changed_cells
                        cell_ctypes = cell_conflict_types.get(cid, set())

                        # Conflict severity for border color (keep band fill always)
                        _ctype_colors = {
                            'Collision': ('#c0392b', 4.0),   # red — critical
                            'Confusion': ('#e67e22', 3.5),   # orange — high
                            'RSI':       ('#8e44ad', 3.5),   # purple — high
                            'Mod3':      ('#f1c40f', 3.0),   # yellow — medium
                            'Mod4':      ('#6d4c41', 2.8),   # brown — medium (NR SSB DMRS)
                            'Mod6':      ('#e74c3c', 2.5),   # light red — low
                            'Mod30':     ('#95a5a6', 2.0),   # gray — low
                        }
                        _ctype_priority = ['Collision','Confusion','RSI','Mod3','Mod4','Mod6','Mod30']

                        # Color logic: always keep band fill color; use border to indicate status
                        fill_color = cfg['hex_fill']
                        border_color = cfg['hex']
                        border_weight = 1.5
                        border_dash = cfg['dash'] or ''
                        fill_opa = 0.25
                        if is_conflict and not is_resolved:
                            # Pick border color from highest-severity conflict type
                            for _ct in _ctype_priority:
                                if _ct in cell_ctypes:
                                    border_color, border_weight = _ctype_colors[_ct]
                                    break
                            border_dash = '8 4'
                            fill_opa = 0.35
                        elif is_resolved:
                            border_color = '#27ae60'
                            border_weight = 3
                            border_dash = ''
                            fill_opa = 0.30
                        elif is_changed:
                            border_color = '#2980b9'
                            border_weight = 2.5
                            border_dash = '6 3'

                        # Build conflict type badges for popup
                        _ctype_badge_colors = {
                            'Collision': '#c0392b', 'Confusion': '#e67e22', 'RSI': '#8e44ad',
                            'Mod3': '#d4ac0d', 'Mod4': '#6d4c41', 'Mod6': '#e74c3c', 'Mod30': '#95a5a6',
                        }
                        conflict_badges = ''
                        if cell_ctypes and not is_resolved:
                            badges = ''.join([
                                f'<span style="background:{_ctype_badge_colors.get(ct,"#888")};color:#fff;'
                                f'padding:1px 6px;border-radius:3px;margin-right:3px;font-size:10px">{ct}</span>'
                                for ct in _ctype_priority if ct in cell_ctypes
                            ])
                            conflict_badges = f'<tr><td colspan="2" style="padding:4px 4px">{badges}</td></tr>'

                        # Build comparison info for popup
                        comp_html = ''
                        if is_changed:
                            o_pci = orig_pci.get(cid)
                            o_rsi = orig_rsi.get(cid)
                            o_pci_s = int(o_pci) if o_pci is not None and not pd.isna(o_pci) else '?'
                            o_rsi_s = int(o_rsi) if o_rsi is not None and not pd.isna(o_rsi) else '?'
                            pci_s = int(pci) if pci is not None and not pd.isna(pci) else '?'
                            rsi_s = int(rsi) if rsi is not None and not pd.isna(rsi) else '?'
                            status = '✅ Çözüldü' if is_resolved else '🔄 Değişti'
                            comp_html = f"""<tr style='background:#e8f5e9'><td style='padding:2px 4px;color:#2e7d32;font-weight:bold' colspan='2'>
  {status} | PCI: {o_pci_s}→{pci_s} | RSI: {o_rsi_s}→{rsi_s}</td></tr>"""
                        elif is_conflict:
                            comp_html = f'<tr><td style="padding:2px 4px;color:#c0392b;font-weight:bold" colspan="2">⚠️ ÇAKIŞMA TESPİT EDİLDİ</td></tr>{conflict_badges}'

                        popup_html = f"""<div style='font-family:Segoe UI,Arial;font-size:12px;width:290px;line-height:1.5'>
<div style='background:{cfg["hex"]};color:white;padding:6px 10px;border-radius:6px 6px 0 0;font-weight:bold;font-size:13px'>
  📶 {bl} — {cid} <span style='font-size:10px;opacity:0.8'>({source_label})</span>
</div>
<div style='padding:8px 10px;border:1px solid #ddd;border-top:none;border-radius:0 0 6px 6px'>
<table style='width:100%;border-collapse:collapse;font-size:11px'>
{comp_html}
<tr><td style='padding:2px 4px;color:#666'>📡 Band</td><td style='padding:2px 4px;font-weight:bold'>{band_mhz} MHz{f" / {int(bw_mhz)} MHz BW" if pd.notna(bw_mhz) and bw_mhz else ""}</td></tr>
<tr style='background:#f8f8f8'><td style='padding:2px 4px;color:#666'>🔢 PCI</td><td style='padding:2px 4px;font-weight:bold'>{int(pci) if pci is not None and not pd.isna(pci) else "N/A"}</td></tr>
<tr><td style='padding:2px 4px;color:#666'>↳ PSS / SSS</td><td style='padding:2px 4px'>{pss} / {sss}</td></tr>
<tr style='background:#f8f8f8'><td style='padding:2px 4px;color:#666'>📻 RSI</td><td style='padding:2px 4px'>{int(rsi) if rsi is not None and not pd.isna(rsi) else "N/A"}</td></tr>
<tr><td style='padding:2px 4px;color:#666'>🧭 Azimuth</td><td style='padding:2px 4px'>{az}°</td></tr>
<tr style='background:#f8f8f8'><td style='padding:2px 4px;color:#666'>📐 Hüzme</td><td style='padding:2px 4px'>{bw}°</td></tr>
<tr><td style='padding:2px 4px;color:#666'>🏢 Site</td><td style='padding:2px 4px'>{site_id}</td></tr>
<tr style='background:#f8f8f8'><td style='padding:2px 4px;color:#666'>📍 Sektör</td><td style='padding:2px 4px'>{sec_key}</td></tr>
<tr><td style='padding:2px 4px;color:#666'>🌐 Koord</td><td style='padding:2px 4px'>{lat:.6f}, {lon:.6f}</td></tr>
</table>
</div></div>"""

                        _tt_conflict = ''
                        if cell_ctypes and not is_resolved:
                            _tt_conflict = ' | ⚠️' + '+'.join(sorted(cell_ctypes))
                        tooltip = f"{cid} | {bl} | PCI:{int(pci) if pci is not None and not pd.isna(pci) else '?'} | Az:{az}°{_tt_conflict}"

                        # Draw sector polygon
                        if pd.notna(az):
                            half_bw = float(bw) / 2
                            range_km = (base_sector_m * cfg['coverage_scale']) / 1000.0
                            pts = [[lat, lon]]
                            for ang_off in np.linspace(-half_bw, half_bw, 16):
                                angle = np.radians(float(az) + ang_off)
                                dlat = (range_km / 111.32) * np.cos(angle)
                                dlon = (range_km / (111.32 * np.cos(np.radians(lat)))) * np.sin(angle)
                                pts.append([lat + dlat, lon + dlon])
                            pts.append([lat, lon])

                            folium.Polygon(
                                pts, color=border_color, fill=True,
                                fill_color=fill_color,
                                fill_opacity=fill_opa, weight=border_weight,
                                dash_array=border_dash,
                                popup=folium.Popup(popup_html, max_width=310),
                                tooltip=tooltip
                            ).add_to(fg)

                            arr_km = range_km * 1.05
                            ang = np.radians(float(az))
                            dlat = (arr_km / 111.32) * np.cos(ang)
                            dlon = (arr_km / (111.32 * np.cos(np.radians(lat)))) * np.sin(ang)
                            folium.PolyLine(
                                [[lat, lon], [lat + dlat, lon + dlon]],
                                color=border_color, weight=2, opacity=0.6,
                                dash_array='6'
                            ).add_to(fg)

                        # Site marker (one per site)
                        site_key = f"{site_id}_{lat:.5f}_{lon:.5f}" if site_id else f"{lat:.5f}_{lon:.5f}"
                        if site_key not in site_markers_placed:
                            site_markers_placed.add(site_key)
                            site_cells = df_orig[df_orig['site_id'] == site_id] if site_id else df_orig[(df_orig['latitude'] == lat) & (df_orig['longitude'] == lon)]

                            site_popup_rows = ""
                            for _, sc in site_cells.iterrows():
                                sc_cid = str(sc['cell_id'])
                                sc_bl = sc.get('band_label', '?')
                                sc_pci = eff_pci.get(sc_cid)
                                sc_rsi = eff_rsi.get(sc_cid)
                                sc_pci_s = int(sc_pci) if sc_pci is not None and not pd.isna(sc_pci) else '?'
                                sc_rsi_s = int(sc_rsi) if sc_rsi is not None and not pd.isna(sc_rsi) else '?'
                                sc_az = sc.get('azimuth', '?')
                                sc_cfg = band_config.get(sc_bl, band_config['Bilinmeyen'])
                                icon = ''
                                if sc_cid in conflict_cells and sc_cid not in resolved_cells:
                                    icon = ' ⚠️'
                                elif sc_cid in resolved_cells:
                                    icon = ' ✅'
                                elif sc_cid in changed_cells:
                                    icon = ' 🔄'
                                site_popup_rows += f"<tr><td style='padding:3px 6px'><span style='color:{sc_cfg['hex']}'>●</span> {sc_cid}{icon}</td><td style='padding:3px 6px'>{sc_bl}</td><td style='padding:3px 6px;text-align:center'>{sc_pci_s}</td><td style='padding:3px 6px;text-align:center'>{sc_rsi_s}</td><td style='padding:3px 6px;text-align:center'>{sc_az}°</td></tr>\n"

                            n_conflict = sum(1 for _, sc in site_cells.iterrows() if str(sc['cell_id']) in conflict_cells and str(sc['cell_id']) not in resolved_cells)
                            n_resolved = sum(1 for _, sc in site_cells.iterrows() if str(sc['cell_id']) in resolved_cells)

                            site_popup = f"""<div style='font-family:Segoe UI,Arial;font-size:12px;width:420px'>
<div style='background:#37474F;color:white;padding:6px 10px;border-radius:6px 6px 0 0;font-weight:bold'>
  🗼 {site_id or 'Site'} — {len(site_cells)} Hücre <span style='font-size:10px;opacity:0.8'>({source_label})</span>
</div>
<div style='padding:6px;border:1px solid #ddd;border-top:none;border-radius:0 0 6px 6px'>
<table style='width:100%;border-collapse:collapse;font-size:11px'>
<tr style='background:#ECEFF1;font-weight:bold'>
<td style='padding:3px 6px'>Cell ID</td><td style='padding:3px 6px'>Band</td>
<td style='padding:3px 6px;text-align:center'>PCI</td><td style='padding:3px 6px;text-align:center'>RSI</td>
<td style='padding:3px 6px;text-align:center'>Az</td>
</tr>
{site_popup_rows}
</table>
<div style='margin-top:4px;font-size:10px;color:#888'>📍 {lat:.6f}, {lon:.6f}{"  |  ⚠️" + str(n_conflict) + " çakışma" if n_conflict else ""}{"  |  ✅" + str(n_resolved) + " çözüldü" if n_resolved else ""}</div>
</div></div>"""

                            has_conflict = n_conflict > 0
                            folium.Marker(
                                [lat, lon],
                                popup=folium.Popup(site_popup, max_width=440),
                                tooltip=f"🗼 {site_id or 'Site'} | {len(site_cells)} hücre",
                                icon=folium.Icon(
                                    color='darkred' if has_conflict else 'darkblue',
                                    icon='tower-broadcast' if not has_conflict else 'triangle-exclamation',
                                    prefix='fa')
                            ).add_to(site_fg)

                # --- Per-conflict-type pair mapping & FeatureGroups ---
                _ctype_line_colors = {
                    'Collision': '#c0392b', 'Confusion': '#e67e22', 'RSI': '#8e44ad',
                    'Mod3': '#d4ac0d', 'Mod4': '#6d4c41', 'Mod6': '#e74c3c', 'Mod30': '#95a5a6',
                }
                _ctype_labels_tr = {
                    'Collision': 'Collision — Aynı PCI',
                    'Confusion': 'Confusion — Karışıklık',
                    'RSI': 'RSI — Root Sequence',
                    'Mod3': 'Mod3 — PSS Girişimi',
                    'Mod4': 'Mod4 — SSB DMRS (NR)',
                    'Mod6': 'Mod6 — RS Çakışması',
                    'Mod30': 'Mod30 — Mod30 Çakışması',
                }

                # Per-type pair counts from already-computed _conflict_tables_data
                conflict_type_counts = defaultdict(int)
                _ctype_order = ['Collision', 'Confusion', 'RSI', 'Mod3', 'Mod4', 'Mod6', 'Mod30']
                if show_conflicts:
                    for clabel, tbl in _conflict_tables_data.items():
                        if clabel not in selected_conflict_types:
                            continue
                        if len(tbl) > 0:
                            conflict_type_counts[clabel] = len(tbl)

                # Cell location lookup
                loc = {str(r['cell_id']): (r['latitude'], r['longitude']) for _, r in df_orig.iterrows()}

                # --- Conflict FeatureGroups (toggleable per type via LayerControl) ---
                if show_conflicts:
                    for ctype in _ctype_order:
                        cnt = conflict_type_counts.get(ctype, 0)
                        if cnt == 0:
                            continue
                        clr = _ctype_line_colors.get(ctype, '#888')
                        fg_c = folium.FeatureGroup(
                            name=f"⚠️ {ctype} ({cnt})", show=(ctype in ('Collision','RSI')))
                        fg_c.add_to(m)
                        # Collect cells involved in this conflict type
                        tbl = _conflict_tables_data.get(ctype, pd.DataFrame())
                        if len(tbl) > 0:
                            _seen_cells = set()
                            for _, crow in tbl.iterrows():
                                for ckey in ('cell_1', 'cell_2'):
                                    cid = str(crow[ckey])
                                    if cid in _seen_cells:
                                        continue
                                    _seen_cells.add(cid)
                                    cl = loc.get(cid)
                                    if cl:
                                        folium.CircleMarker(
                                            cl, radius=8, color=clr,
                                            fill=True, fill_color=clr,
                                            fill_opacity=0.6, weight=2,
                                            tooltip=f"⚠️ {ctype}: {cid}"
                                        ).add_to(fg_c)

                # Neighbor lines (optional, non-conflict visualization)
                if show_neighbors and results:
                    nb_group = folium.FeatureGroup(name="🔗 Komşuluk Hatları", show=True)
                    nb_group.add_to(m)
                    for c, ns in results['neighbors'].items():
                        for nb in ns:
                            cl_nb, nl_nb = loc.get(str(c)), loc.get(str(nb))
                            if cl_nb and nl_nb:
                                folium.PolyLine([cl_nb, nl_nb], color='#aaa',
                                                weight=1, opacity=0.3).add_to(nb_group)

                folium.LayerControl(collapsed=False).add_to(m)

                # Legend
                legend_items = ''.join([
                    f'<div style="display:flex;align-items:center;margin:2px 0">'
                    f'<span style="display:inline-block;width:14px;height:14px;background:{band_config.get(bl, band_config["Bilinmeyen"])["hex_fill"]};'
                    f'border:2px solid {band_config.get(bl, band_config["Bilinmeyen"])["hex"]};border-radius:3px;margin-right:6px"></span>'
                    f' {bl} (×{band_config.get(bl, band_config["Bilinmeyen"])["coverage_scale"]})</div>'
                    for bl in band_labels_present
                ])
                # Build conflict legend items with counts (only selected types)
                conflict_legend_items = ''
                for ctype in _ctype_order:
                    if ctype not in selected_conflict_types:
                        continue
                    cnt = conflict_type_counts.get(ctype, 0)
                    clr = _ctype_line_colors.get(ctype, '#888')
                    lbl_tr = _ctype_labels_tr.get(ctype, ctype)
                    cnt_badge = f' <span style="background:{clr};color:#fff;padding:0 5px;border-radius:8px;font-size:9px;font-weight:bold">{cnt}</span>' if cnt > 0 else ''
                    conflict_legend_items += (
                        f'<div style="display:flex;align-items:center;margin:2px 0">'
                        f'<span style="display:inline-block;width:14px;height:14px;'
                        f'border:3px dashed {clr};border-radius:3px;margin-right:6px"></span>'
                        f' <b>{ctype}</b> — {lbl_tr.split("—")[-1].strip()}{cnt_badge}</div>'
                    )
                legend = f"""
                <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                            background:white;padding:12px 16px;border-radius:8px;
                            border:2px solid #ccc;font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
                            max-height:70vh;overflow-y:auto">
                    <b>📶 {source_label}</b><br>
                    <span style="font-size:10px;color:#666">Büyük dilim = geniş kapsama</span><br>
                    <hr style="margin:4px 0">
                    {legend_items}
                    <hr style="margin:4px 0">
                    <b style="font-size:11px">Çakışma Türleri (sektör kenarı):</b><br>
                    {conflict_legend_items}
                    <hr style="margin:4px 0">
                    <div style="margin:2px 0"><span style="display:inline-block;width:14px;height:3px;background:#27ae60;margin-right:6px"></span> ✅ Çözüldü</div>
                    <div style="margin:2px 0"><span style="display:inline-block;width:14px;height:3px;border-top:2px dashed #2980b9;margin-right:6px"></span> 🔄 Değişti</div>
                </div>"""
                m.get_root().html.add_child(folium.Element(legend))

                # Save map to temp file (full standalone HTML)
                map_tmp = os.path.join(tempfile.gettempdir(), 'pci_rsi_map.html')
                m.save(map_tmp)
                with open(map_tmp, 'r', encoding='utf-8') as f:
                    raw_html = f.read()
                # Inject CSS so map fills the iframe properly
                height_css = '<style>html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}</style>'
                raw_html = raw_html.replace('<head>', '<head>' + height_css, 1)
                st.session_state.map_html = raw_html
            st.success(f"✅ Harita oluşturuldu! ({source_label})")

        # Display map from session state
        if st.session_state.map_html:
            # Render map HTML directly via st.components (avoids base64 size limits)
            st.components.v1.html(st.session_state.map_html, height=720, scrolling=False)

            st.download_button("🗺️ Haritayı HTML Olarak İndir",
                               st.session_state.map_html.encode('utf-8'),
                               "pci_rsi_harita.html", "text/html",
                               use_container_width=True)
        else:
            st.info("📌 Haritayı görmek için yukarıdaki butona tıklayın.")

# ============================================================
# TAB 4 — DETAILED REPORTS
# ============================================================
with tab4:
    if st.session_state.results is None:
        st.info("📌 Önce analizi başlatın.")
    else:
        _stale_results_warning()
        results = st.session_state.results

        # Helper: strip co-sector pairs from conflict DataFrames (safety-net)
        def _strip_co_sector(conflict_df):
            """Remove rows where cell_1 and cell_2 are co-sector.
            Vectorized: builds a set of co-sector pairs for O(1) lookup."""
            if conflict_df is None or len(conflict_df) == 0:
                return conflict_df
            c2s = st.session_state.cell_to_sector or {}
            # Build co-sector pair set from cell_to_sector dict
            _co_pairs = set()
            if c2s:
                from itertools import combinations
                _sector_members = defaultdict(list)
                for cid, sk in c2s.items():
                    if sk is not None:
                        _sector_members[sk].append(str(cid))
                for members in _sector_members.values():
                    if len(members) > 1:
                        for a, b in combinations(members, 2):
                            _co_pairs.add((a, b) if a < b else (b, a))

            c1_arr = conflict_df['cell_1'].astype(str).values
            c2_arr = conflict_df['cell_2'].astype(str).values
            keep = []
            for i in range(len(c1_arr)):
                a, b = c1_arr[i], c2_arr[i]
                if _is_co_sector_by_id(a, b):
                    keep.append(False)
                    continue
                pair = (a, b) if a < b else (b, a)
                if pair in _co_pairs:
                    keep.append(False)
                    continue
                keep.append(True)
            return conflict_df[keep].reset_index(drop=True)

        rtype = st.selectbox("Rapor Tipi", [
            "🔴 PCI Collision","🟠 PCI Confusion","🟡 Mod 3 Conflict",
            "🟤 Mod 4 Conflict (NR SSB DMRS)",
            "🔵 Mod 6 Conflict","⚪ Mod 30 Conflict",
            "🏠 Co-site Collision & Mod3/Mod4",
            "🟣 RSI Collision (Cell-Range)","📡 PRACH / Cell Range Bilgileri","🔗 Komşuluk Tablosu"])
        st.markdown("---")

        # Helper: filter to only co-site pairs (same site, DIFFERENT sector)
        def _filter_cosite(conflict_df):
            """Keep only rows where cell_1 and cell_2 are co-site but NOT co-sector.
            Vectorized with array iteration instead of df.apply."""
            if conflict_df is None or len(conflict_df) == 0:
                return pd.DataFrame()
            df_src = st.session_state.df
            _sm = dict(zip(df_src['cell_id'].astype(str), df_src['site_id'].astype(str))) if df_src is not None and 'site_id' in df_src.columns else {}
            _c2s = st.session_state.cell_to_sector or {}
            c1_arr = conflict_df['cell_1'].astype(str).values
            c2_arr = conflict_df['cell_2'].astype(str).values
            keep = []
            for i in range(len(c1_arr)):
                c1, c2 = c1_arr[i], c2_arr[i]
                # Exclude co-sector pairs
                if _is_co_sector_by_id(c1, c2):
                    keep.append(False)
                    continue
                if _c2s:
                    sa, sb = _c2s.get(c1), _c2s.get(c2)
                    if sa is not None and sa == sb:
                        keep.append(False)
                        continue
                # Check same site
                if _sm:
                    s1, s2 = _sm.get(c1), _sm.get(c2)
                    if s1 and s2 and s1 == s2 and s1 not in ('', 'nan', None):
                        keep.append(True)
                        continue
                keep.append(_is_same_site_by_id(c1, c2))
            return conflict_df[keep].reset_index(drop=True)

        if "Collision" in rtype and "RSI" not in rtype and "Co-site" not in rtype:
            st.markdown("""### 🔴 PCI Collision
> Komşu hücrelerde aynı PCI → UE senkronizasyon başarısızlığı. **KRİTİK!**""")
            d = _strip_co_sector(results['collisions'])
            if len(d)>0: st.dataframe(d, use_container_width=True, height=400)
            else: st.success("✅ PCI Collision bulunamadı!")

        elif "Confusion" in rtype:
            st.markdown("""### 🟠 PCI Confusion
> Bir hücrenin iki komşusu aynı PCI → handover hedef belirsizliği. **YÜKSEK**""")
            d = _strip_co_sector(results['confusions'])
            if len(d)>0: st.dataframe(d, use_container_width=True, height=400)
            else: st.success("✅ PCI Confusion bulunamadı!")

        elif "Mod 3" in rtype:
            st.markdown("""### 🟡 Mod 3 Conflict
> Komşularda PCI mod 3 aynı → PSS interferansı""")
            d = _strip_co_sector(results['mod3_conflicts'])
            if len(d)>0: st.dataframe(d, use_container_width=True, height=400)
            else: st.success("✅ Mod 3 bulunamadı!")

        elif "Mod 4" in rtype:
            st.markdown("""### 🟤 Mod 4 Conflict (NR SSB DMRS)
> 5G NR'da komşu hücrelerde PCI mod 4 aynı → SSB DMRS interferansı.
> Bu kontrol yalnızca NR (5G) teknolojisi için geçerlidir.
> (3GPP TS 38.211 — SSB DMRS sekansı PCI mod 4'e bağlıdır)""")
            d = _strip_co_sector(results.get('mod4_conflicts', pd.DataFrame()))
            if len(d)>0: st.dataframe(d, use_container_width=True, height=400)
            else: st.success("✅ Mod 4 bulunamadı!")

        elif "Mod 6" in rtype:
            st.markdown("""### 🔵 Mod 6 Conflict
> PCI mod 6 aynı → RS frekans kaydırma çakışması""")
            d = _strip_co_sector(results['mod6_conflicts'])
            if len(d)>0: st.dataframe(d, use_container_width=True, height=400)
            else: st.success("✅ Mod 6 bulunamadı!")

        elif "Mod 30" in rtype:
            st.markdown("""### ⚪ Mod 30 Conflict""")
            d = _strip_co_sector(results['mod30_conflicts'])
            if len(d)>0: st.dataframe(d, use_container_width=True, height=400)
            else: st.success("✅ Mod 30 bulunamadı!")

        elif "Co-site" in rtype:
            st.markdown("""### 🏠 Co-site Çakışma Raporu
> **Co-site** = aynı fiziksel site, farklı sektör (aynı azimut DEĞİL).
> Co-site hücrelerinde PCI collision ve mod3/mod4 çakışması **kesinlikle** olmamalıdır.
> Bu rapor yalnızca co-site çiftlerini gösterir.""")

            # Co-site Collisions
            cs_col = _filter_cosite(results['collisions'])
            _n_cs_col = len(cs_col)
            st.markdown(f"#### 🔴 Co-site PCI Collision — **{_n_cs_col}** çift")
            if _n_cs_col > 0:
                st.dataframe(cs_col, use_container_width=True, height=300)
            else:
                st.success("✅ Co-site PCI Collision yok!")

            # Co-site Mod3
            cs_m3 = _filter_cosite(results['mod3_conflicts'])
            _n_cs_m3 = len(cs_m3)
            st.markdown(f"#### 🟡 Co-site Mod3 Çakışması — **{_n_cs_m3}** çift")
            if _n_cs_m3 > 0:
                st.dataframe(cs_m3, use_container_width=True, height=300)
            else:
                st.success("✅ Co-site Mod3 çakışması yok!")

            # Co-site Mod4 (NR)
            cs_m4 = _filter_cosite(results.get('mod4_conflicts', pd.DataFrame()))
            _n_cs_m4 = len(cs_m4)
            st.markdown(f"#### 🟤 Co-site Mod4 Çakışması (NR SSB DMRS) — **{_n_cs_m4}** çift")
            if _n_cs_m4 > 0:
                st.dataframe(cs_m4, use_container_width=True, height=300)
            else:
                st.success("✅ Co-site Mod4 çakışması yok!")

            # Summary
            _total_cs = _n_cs_col + _n_cs_m3 + _n_cs_m4
            if _total_cs > 0:
                st.error(f"⚠️ Toplam **{_total_cs}** co-site çakışması tespit edildi. "
                         f"Bu çakışmalar ağ performansını ciddi şekilde etkiler ve "
                         f"**Otomatik PCI Planlayıcı** ile çözülebilir.")
            else:
                st.success("🎉 Hiçbir co-site çakışması yok — harika!")

        elif "RSI" in rtype:
            st.markdown("""### 🟣 RSI Collision (Cell-Range-Aware)
> RSI çakışması **PRACH Ncs → root sequence sayısı** baz alınarak hesaplanır.
> Her hücre `ceil(64 / floor(Nzc/Ncs))` kadar ardışık root sequence kullanır.
> Komşu hücrelerin RSI aralıkları çakışırsa → PRACH preamble çakışması. **YÜKSEK**""")
            d = _strip_co_sector(results['rsi_collisions'])
            if len(d)>0:
                st.dataframe(d, use_container_width=True, height=400)
                st.markdown("#### Ncs → Root Sequence Hesaplama")
                st.markdown("""
| zeroCorrelationZoneConfig | Ncs | Preambles/Root | Roots Needed (64 pre) | Cell Range (km) |
|:---:|:---:|:---:|:---:|:---:|""" + "\n".join([
    f"| {k} | {v} | {max(1,839//v) if v>0 else 1} | {int(np.ceil(64/(max(1,839//v) if v>0 else 1)))} | {round(cell_range_from_ncs(v),2)} |"
    for k,v in sorted(LTE_NCS_UNRESTRICTED.items()) if v > 0]))
            else:
                st.success("✅ RSI Collision bulunamadı!")

        elif "PRACH" in rtype:
            st.markdown("""### 📡 PRACH / Cell Range Bilgileri
> **PRACH Config Index** → Preamble Format (3GPP TS 36.211 Table 5.7.1-2)
> **zeroCorrelationZoneConfig** → Ncs (3GPP TS 36.211 Table 5.7.2-2)
> **Cell Range (Ncs)** = (Ncs / 839) × Tseq × c / 2
> **Roots Needed** = ⌈64 / ⌊839 / Ncs⌋⌉""")
            pi = st.session_state.prach_info
            if pi is not None:
                st.dataframe(pi, use_container_width=True, height=400)
            # Reference tables
            with st.expander("📋 3GPP Ncs Referans Tablosu (Unrestricted)"):
                ref = pd.DataFrame([
                    {'Config': k, 'Ncs': v,
                     'Preambles/Root': max(1, 839//v) if v>0 else 1,
                     'Roots (64)': int(np.ceil(64/(max(1,839//v) if v>0 else 1))),
                     'Cell Range (km)': round(cell_range_from_ncs(v), 2)}
                    for k,v in sorted(LTE_NCS_UNRESTRICTED.items())])
                st.table(ref)
            with st.expander("📋 Preamble Format → Max Cell Range"):
                fref = pd.DataFrame([
                    {'Format': k, 'Tcp (µs)': v['tcp_us'], 'Tseq (µs)': v['tseq_us'],
                     'Max Range (km)': round(cell_range_from_format(k), 2)}
                    for k,v in sorted(LTE_PREAMBLE_FORMATS.items())])
                st.table(fref)

        elif "Komşuluk" in rtype:
            st.markdown("### 🔗 Komşuluk Tablosu")
            nt = st.session_state.neighbor_table
            if nt is not None and len(nt)>0:
                st.markdown(f"**{len(nt)}** komşuluk ilişkisi")
                st.dataframe(nt, use_container_width=True, height=400)
            else:
                st.warning("Komşuluk bulunamadı.")

        # ==============================================================
        # POST-PLAN DETAIL REPORTS — comparison with current state
        # ==============================================================
        _pci_plan_rpt = st.session_state.pci_plan
        if _pci_plan_rpt is not None and len(_pci_plan_rpt) > 0:
            st.markdown("---")
            st.markdown("## 📊 Plan Sonrası Detaylı Kıyaslama Raporu")
            st.markdown("> Otomatik PCI planının mevcut durumla karşılaştırması. "
                        "Tüm çakışma türleri hem **mevcut** hem de **plan sonrası** değerlerle gösterilir.")

            # Build override key for caching
            _pci_over_rpt = {}
            for _, r in _pci_plan_rpt.iterrows():
                sv = r.get('planned_pci', '—')
                if str(sv) not in ('—', 'nan', 'None'):
                    _pci_over_rpt[str(r['cell_id'])] = int(float(sv))
            _detail_cache_key = frozenset(_pci_over_rpt.items())

            # Check if we already have cached results
            _cached = st.session_state._plan_detail_cache
            _tech_rpt = results['summary']['technology']
            if _cached is not None and _cached.get('_key') == _detail_cache_key:
                _plan_col = _cached['col']
                _plan_con = _cached['con']
                _plan_m3 = _cached['m3']
                _plan_m4 = _cached['m4']
                _plan_m6 = _cached['m6']
                _plan_m30 = _cached['m30']
                _plan_rsi = _cached['rsi']
            else:
                # Build planned DataFrame by overriding PCIs
                _df_plan = st.session_state.df.copy()
                _cid_to_idx_rpt = {}
                for idx, cid_val in _df_plan['cell_id'].items():
                    _cid_to_idx_rpt[str(cid_val)] = idx
                for cid, pval in _pci_over_rpt.items():
                    idx = _cid_to_idx_rpt.get(cid)
                    if idx is not None:
                        _df_plan.at[idx, 'pci'] = pval

                nb = results['neighbors']
                c2s = st.session_state.cell_to_sector or {}
                _na_rpt = results.get('neighbor_attempts', {})
                _lac_rpt = _build_loc_az_maps(_df_plan)

                # Detect conflicts for planned state
                _plan_col = detect_collisions(_df_plan, nb, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt)
                _plan_con = detect_confusions(_df_plan, nb, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt)
                _plan_m3 = detect_mod3_conflicts(_df_plan, nb, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt) if check_mod3 else pd.DataFrame()
                _plan_m4 = detect_mod4_conflicts(_df_plan, nb, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt) if check_mod4 else pd.DataFrame()
                _plan_m6 = detect_mod6_conflicts(_df_plan, nb, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt) if check_mod6 else pd.DataFrame()
                _plan_m30 = detect_mod30_conflicts(_df_plan, nb, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt) if check_mod30 else pd.DataFrame()
                _plan_rsi = detect_rsi_collisions(_df_plan, nb, 'rsi', _tech_rpt, cell_to_sector=c2s, nbr_attempts=_na_rpt, _loc_az_cache=_lac_rpt) if (check_rsi and 'rsi' in _df_plan.columns) else pd.DataFrame()

                st.session_state._plan_detail_cache = {
                    '_key': _detail_cache_key,
                    'col': _plan_col, 'con': _plan_con, 'm3': _plan_m3,
                    'm4': _plan_m4, 'm6': _plan_m6, 'm30': _plan_m30, 'rsi': _plan_rsi,
                }

            # Co-site counts for planned
            _df_src = st.session_state.df
            _sm_rpt = dict(zip(_df_src['cell_id'].astype(str), _df_src['site_id'].astype(str))) if _df_src is not None and 'site_id' in _df_src.columns else {}
            def _cs_cnt_rpt(cdf):
                if len(cdf) == 0: return 0
                n = 0
                for _, rr in cdf.iterrows():
                    a, b = str(rr['cell_1']), str(rr['cell_2'])
                    same = False
                    if _sm_rpt:
                        same = (_sm_rpt.get(a) == _sm_rpt.get(b) and _sm_rpt.get(a) not in (None, '', 'nan'))
                    if not same:
                        same = _is_same_site_by_id(a, b)
                    if same:
                        n += 1
                return n

            # Current state counts
            _cur_col = _strip_co_sector(results['collisions'])
            _cur_con = _strip_co_sector(results['confusions'])
            _cur_m3 = _strip_co_sector(results['mod3_conflicts'])
            _cur_m4 = _strip_co_sector(results.get('mod4_conflicts', pd.DataFrame()))
            _cur_m6 = _strip_co_sector(results['mod6_conflicts'])
            _cur_m30 = _strip_co_sector(results['mod30_conflicts'])
            _cur_rsi = _strip_co_sector(results['rsi_collisions'])

            _plan_col_s = _strip_co_sector(_plan_col)
            _plan_con_s = _strip_co_sector(_plan_con)
            _plan_m3_s = _strip_co_sector(_plan_m3)
            _plan_m4_s = _strip_co_sector(_plan_m4)
            _plan_m6_s = _strip_co_sector(_plan_m6)
            _plan_m30_s = _strip_co_sector(_plan_m30)
            _plan_rsi_s = _strip_co_sector(_plan_rsi)

            # Co-site counts
            _cs_col_cur = results['summary'].get('cosite_collision_count', 0)
            _cs_m3_cur = results['summary'].get('cosite_mod3_count', 0)
            _cs_m4_cur = results['summary'].get('cosite_mod4_count', 0)
            _cs_col_plan = _cs_cnt_rpt(_plan_col)
            _cs_m3_plan = _cs_cnt_rpt(_plan_m3)
            _cs_m4_plan = _cs_cnt_rpt(_plan_m4)

            # Summary comparison table
            _comp_rows = [
                {'Çakışma Türü': '🔴 PCI Collision', 'Mevcut': len(_cur_col), 'Plan Sonrası': len(_plan_col_s),
                 'Fark': len(_plan_col_s) - len(_cur_col)},
                {'Çakışma Türü': '  ↳ Co-site Collision', 'Mevcut': _cs_col_cur, 'Plan Sonrası': _cs_col_plan,
                 'Fark': _cs_col_plan - _cs_col_cur},
                {'Çakışma Türü': '🟠 PCI Confusion', 'Mevcut': len(_cur_con), 'Plan Sonrası': len(_plan_con_s),
                 'Fark': len(_plan_con_s) - len(_cur_con)},
                {'Çakışma Türü': '🟡 Mod 3 Conflict', 'Mevcut': len(_cur_m3), 'Plan Sonrası': len(_plan_m3_s),
                 'Fark': len(_plan_m3_s) - len(_cur_m3)},
                {'Çakışma Türü': '  ↳ Co-site Mod3', 'Mevcut': _cs_m3_cur, 'Plan Sonrası': _cs_m3_plan,
                 'Fark': _cs_m3_plan - _cs_m3_cur},
            ]
            if _tech_rpt == 'NR':
                _comp_rows.append({'Çakışma Türü': '🟤 Mod 4 Conflict (NR)', 'Mevcut': len(_cur_m4), 'Plan Sonrası': len(_plan_m4_s),
                                   'Fark': len(_plan_m4_s) - len(_cur_m4)})
                _comp_rows.append({'Çakışma Türü': '  ↳ Co-site Mod4', 'Mevcut': _cs_m4_cur, 'Plan Sonrası': _cs_m4_plan,
                                   'Fark': _cs_m4_plan - _cs_m4_cur})
            else:
                _comp_rows.append({'Çakışma Türü': '🔵 Mod 6 Conflict', 'Mevcut': len(_cur_m6), 'Plan Sonrası': len(_plan_m6_s),
                                   'Fark': len(_plan_m6_s) - len(_cur_m6)})
                _comp_rows.append({'Çakışma Türü': '⚪ Mod 30 Conflict', 'Mevcut': len(_cur_m30), 'Plan Sonrası': len(_plan_m30_s),
                                   'Fark': len(_plan_m30_s) - len(_cur_m30)})
            _comp_rows.append({'Çakışma Türü': '🟣 RSI Collision', 'Mevcut': len(_cur_rsi), 'Plan Sonrası': len(_plan_rsi_s),
                               'Fark': len(_plan_rsi_s) - len(_cur_rsi)})

            _comp_df = pd.DataFrame(_comp_rows)

            # Style the comparison table
            def _style_diff(val):
                if val < 0:
                    return 'color: #2e7d32; font-weight: bold'  # green = improvement
                elif val > 0:
                    return 'color: #c62828; font-weight: bold'  # red = worse
                return 'color: #666'

            styled_comp = _comp_df.style.map(_style_diff, subset=['Fark'])
            st.dataframe(styled_comp, use_container_width=True, hide_index=True)

            # Improvement summary
            _total_cur = len(_cur_col) + len(_cur_con) + len(_cur_m3) + len(_cur_m6) + len(_cur_m30) + len(_cur_rsi) + len(_cur_m4)
            _total_plan = len(_plan_col_s) + len(_plan_con_s) + len(_plan_m3_s) + len(_plan_m6_s) + len(_plan_m30_s) + len(_plan_rsi_s) + len(_plan_m4_s)
            _improvement = _total_cur - _total_plan
            if _improvement > 0:
                st.success(f"🎉 Plan toplam **{_improvement}** çakışmayı giderdi! ({_total_cur} → {_total_plan})")
            elif _improvement == 0:
                st.info("ℹ️ Toplam çakışma sayısı değişmedi.")
            else:
                st.warning(f"⚠️ Plan sonrası toplam çakışma **{abs(_improvement)}** arttı. ({_total_cur} → {_total_plan})")

            # Detailed tables per conflict type (expandable)
            with st.expander("🔴 Plan Sonrası — PCI Collision Detayları", expanded=False):
                if len(_plan_col_s) > 0:
                    st.dataframe(_plan_col_s, use_container_width=True, height=300)
                else:
                    st.success("✅ Plan sonrası PCI Collision yok!")

            with st.expander("🟠 Plan Sonrası — PCI Confusion Detayları", expanded=False):
                if len(_plan_con_s) > 0:
                    st.dataframe(_plan_con_s, use_container_width=True, height=300)
                else:
                    st.success("✅ Plan sonrası PCI Confusion yok!")

            with st.expander("🟡 Plan Sonrası — Mod 3 Conflict Detayları", expanded=False):
                if len(_plan_m3_s) > 0:
                    st.dataframe(_plan_m3_s, use_container_width=True, height=300)
                else:
                    st.success("✅ Plan sonrası Mod 3 Conflict yok!")

            with st.expander("🏠 Plan Sonrası — Co-site Çakışma Detayları", expanded=False):
                _plan_cs_col = _filter_cosite(_plan_col)
                _plan_cs_m3 = _filter_cosite(_plan_m3)
                _plan_cs_m4 = _filter_cosite(_plan_m4)
                _n_p_cs = len(_plan_cs_col) + len(_plan_cs_m3) + len(_plan_cs_m4)
                if _n_p_cs == 0:
                    st.success("🎉 Plan sonrası hiçbir co-site çakışması yok!")
                else:
                    if len(_plan_cs_col) > 0:
                        st.markdown(f"**Co-site Collision — {len(_plan_cs_col)} çift:**")
                        st.dataframe(_plan_cs_col, use_container_width=True, height=200)
                    if len(_plan_cs_m3) > 0:
                        st.markdown(f"**Co-site Mod3 — {len(_plan_cs_m3)} çift:**")
                        st.dataframe(_plan_cs_m3, use_container_width=True, height=200)
                    if len(_plan_cs_m4) > 0:
                        st.markdown(f"**Co-site Mod4 — {len(_plan_cs_m4)} çift:**")
                        st.dataframe(_plan_cs_m4, use_container_width=True, height=200)

            if _tech_rpt == 'NR':
                with st.expander("🟤 Plan Sonrası — Mod 4 Conflict Detayları (NR)", expanded=False):
                    if len(_plan_m4_s) > 0:
                        st.dataframe(_plan_m4_s, use_container_width=True, height=300)
                    else:
                        st.success("✅ Plan sonrası Mod 4 Conflict yok!")
            else:
                with st.expander("🔵 Plan Sonrası — Mod 6 Conflict Detayları", expanded=False):
                    if len(_plan_m6_s) > 0:
                        st.dataframe(_plan_m6_s, use_container_width=True, height=300)
                    else:
                        st.success("✅ Plan sonrası Mod 6 Conflict yok!")
                with st.expander("⚪ Plan Sonrası — Mod 30 Conflict Detayları", expanded=False):
                    if len(_plan_m30_s) > 0:
                        st.dataframe(_plan_m30_s, use_container_width=True, height=300)
                    else:
                        st.success("✅ Plan sonrası Mod 30 Conflict yok!")

            with st.expander("🟣 Plan Sonrası — RSI Collision Detayları", expanded=False):
                if len(_plan_rsi_s) > 0:
                    st.dataframe(_plan_rsi_s, use_container_width=True, height=300)
                else:
                    st.success("✅ Plan sonrası RSI Collision yok!")

# ============================================================
# TAB 5 — PCI/RSI SUGGESTIONS + PER-CELL RESCAN
# ============================================================
with tab5:
    if st.session_state.results is None:
        st.info("📌 Önce veri yükleyip analizi başlatın.")
    else:
        _stale_results_warning()
        _sug_results = st.session_state.results
        _sug_df = st.session_state.df
        _sug_nb = _sug_results['neighbors']
        _sug_tech = _sug_results['summary']['technology']
        _sug_sg = st.session_state.sector_groups or {}
        _sug_c2s = st.session_state.cell_to_sector or {}

        # ─── SECTION 1: PCI / RSI Öneriler (Suggestions) ───
        st.markdown("## 💡 PCI / RSI Öneriler")
        st.markdown("""> Mevcut analizde tespit edilen **çakışmalı hücreler** için temiz PCI ve RSI önerileri sunar.
> Sadece sorunlu hücrelere yeni değer önerir — sorunsuz hücreler değiştirilmez.
> Öneriler mevcut ağ üzerinde **en az değişiklik** prensibiyle çalışır.""")

        _sug_c1, _sug_c2 = st.columns(2)
        with _sug_c1:
            if st.button("📡 PCI Önerileri Hesapla", type="primary", use_container_width=True):
                _pci_prog_bar = st.progress(0, text="PCI önerileri hesaplanıyor...")
                try:
                    def _pci_prog(cur, tot):
                        pct = min(cur / max(tot, 1), 1.0)
                        _pci_prog_bar.progress(pct, text=f"PCI önerileri: {cur}/{tot} hücre işleniyor...")
                    _pci_sug = suggest_pci(
                        _sug_df, _sug_nb, _sug_results,
                        technology=_sug_tech,
                        check_mod3=check_mod3, check_mod6=check_mod6,
                        check_mod30=check_mod30,
                        sector_groups=_sug_sg,
                        cell_to_sector=_sug_c2s,
                        nbr_attempts=_sug_results.get('neighbor_attempts', {}),
                        check_mod4=check_mod4,
                        progress_fn=_pci_prog,
                    )
                    _pci_prog_bar.progress(1.0, text="PCI önerileri tamamlandı!")
                    st.session_state['pci_suggestions'] = _pci_sug
                    st.session_state['_sug_cache_key'] = None
                    st.session_state['_sug_cache_result'] = None
                except Exception as e:
                    _pci_prog_bar.empty()
                    st.error(f"❌ PCI önerisi hatası: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        with _sug_c2:
            if st.button("📻 RSI Önerileri Hesapla", type="primary", use_container_width=True):
                _rsi_prog_bar = st.progress(0, text="RSI önerileri hesaplanıyor...")
                try:
                    def _rsi_prog(cur, tot):
                        pct = min(cur / max(tot, 1), 1.0)
                        _rsi_prog_bar.progress(pct, text=f"RSI önerileri: {cur}/{tot} hücre işleniyor...")
                    _rsi_sug = suggest_rsi(
                        _sug_df, _sug_nb, _sug_results,
                        technology=_sug_tech,
                        sector_groups=_sug_sg,
                        cell_to_sector=_sug_c2s,
                        progress_fn=_rsi_prog,
                    )
                    _rsi_prog_bar.progress(1.0, text="RSI önerileri tamamlandı!")
                    st.session_state['rsi_suggestions'] = _rsi_sug
                    st.session_state['_sug_cache_key'] = None
                    st.session_state['_sug_cache_result'] = None
                except Exception as e:
                    _rsi_prog_bar.empty()
                    st.error(f"❌ RSI önerisi hatası: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Show PCI suggestions
        _pci_sug_df = st.session_state.get('pci_suggestions')
        if _pci_sug_df is not None:
            if len(_pci_sug_df) == 0:
                st.success("✅ PCI çakışması tespit edilmedi — öneri gerekmiyor.")
            else:
                st.markdown("### 📡 PCI Önerileri")
                _sug_ok = len(_pci_sug_df[_pci_sug_df['suggested_pci'].astype(str) != '—'])
                _sug_fail = len(_pci_sug_df) - _sug_ok
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Sorunlu Hücre", len(_pci_sug_df))
                sc2.metric("✅ Öneri Bulunan", _sug_ok)
                sc3.metric("❌ Öneri Bulunamayan", _sug_fail)
                st.dataframe(_pci_sug_df, use_container_width=True, height=400)

                _pci_sug_buf = io.BytesIO()
                with pd.ExcelWriter(_pci_sug_buf, engine='xlsxwriter') as w:
                    _pci_sug_df.to_excel(w, sheet_name='PCI Önerileri', index=False)
                st.download_button("📥 PCI Önerilerini Excel İndir", _pci_sug_buf.getvalue(),
                                   "pci_onerileri.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

        # Show RSI suggestions
        _rsi_sug_df = st.session_state.get('rsi_suggestions')
        if _rsi_sug_df is not None:
            if len(_rsi_sug_df) == 0:
                st.success("✅ RSI çakışması tespit edilmedi — öneri gerekmiyor.")
            else:
                st.markdown("### 📻 RSI Önerileri")
                _rsug_ok = len(_rsi_sug_df[_rsi_sug_df['suggested_rsi'].astype(str) != '—'])
                _rsug_fail = len(_rsi_sug_df) - _rsug_ok
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Sorunlu Hücre", len(_rsi_sug_df))
                rc2.metric("✅ Öneri Bulunan", _rsug_ok)
                rc3.metric("❌ Öneri Bulunamayan", _rsug_fail)
                st.dataframe(_rsi_sug_df, use_container_width=True, height=400)

                _rsi_sug_buf = io.BytesIO()
                with pd.ExcelWriter(_rsi_sug_buf, engine='xlsxwriter') as w:
                    _rsi_sug_df.to_excel(w, sheet_name='RSI Önerileri', index=False)
                st.download_button("📥 RSI Önerilerini Excel İndir", _rsi_sug_buf.getvalue(),
                                   "rsi_onerileri.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

        # ─── SECTION 2: Per-Site / Per-Cell PCI/RSI Rescan ───
        st.markdown("---")
        st.markdown("## 🔄 Seçici PCI / RSI Yeniden Tarama")
        st.markdown("""> Ağdaki **belirli saha veya hücreleri** seçerek sadece onlar için yeni PCI ve/veya RSI değeri bulur.
> Seçilmeyen hücreler sabit kalır, yeni değerler mevcut ağa uyumlu olarak atanır.""")

        _rescan_mode = st.radio("Seçim Modu", ["📡 Saha (Site) Bazlı", "🔧 Hücre (Cell) Bazlı"],
                                 horizontal=True, key="rescan_mode")

        _rescan_what = st.multiselect("Ne taransın?",
                                       ["PCI", "RSI"],
                                       default=["PCI", "RSI"],
                                       key="rescan_what")

        _rescan_target_cells = []

        if _rescan_mode == "📡 Saha (Site) Bazlı":
            if 'site_id' in _sug_df.columns:
                _all_sites = sorted(_sug_df['site_id'].dropna().astype(str).unique().tolist())
                _selected_sites = st.multiselect(
                    f"Saha seçin ({len(_all_sites)} saha mevcut)",
                    _all_sites,
                    key="rescan_sites",
                    help="Birden fazla saha seçebilirsiniz."
                )
                if _selected_sites:
                    _site_mask = _sug_df['site_id'].astype(str).isin(_selected_sites)
                    _rescan_target_cells = _sug_df.loc[_site_mask, 'cell_id'].astype(str).tolist()
                    st.info(f"Seçilen sahalar: **{len(_selected_sites)}** → Toplam **{len(_rescan_target_cells)}** hücre taranacak.")
            else:
                st.warning("⚠️ Veri setinde `site_id` kolonu bulunamadı. Hücre bazlı seçim kullanın.")
        else:
            _all_cells = sorted(_sug_df['cell_id'].astype(str).unique().tolist())
            _selected_cells = st.multiselect(
                f"Hücre seçin ({len(_all_cells)} hücre mevcut)",
                _all_cells,
                key="rescan_cells",
                help="Birden fazla hücre seçebilirsiniz."
            )
            _rescan_target_cells = _selected_cells

        if _rescan_target_cells and _rescan_what:
            if st.button("🔍 Seçili Hücreler İçin PCI/RSI Tara", type="primary",
                         use_container_width=True):
                with st.spinner(f"{len(_rescan_target_cells)} hücre için tarama yapılıyor..."):
                    try:
                        _rescan_result = rescan_pci_rsi_for_cells(
                            df=_sug_df,
                            neighbors=_sug_nb,
                            target_cell_ids=_rescan_target_cells,
                            technology=_sug_tech,
                            check_mod3=check_mod3,
                            check_mod6=check_mod6,
                            check_mod30=check_mod30,
                            check_mod4=check_mod4,
                            sector_groups=_sug_sg,
                            cell_to_sector=_sug_c2s,
                            rescan_pci=("PCI" in _rescan_what),
                            rescan_rsi=("RSI" in _rescan_what),
                        )
                        st.session_state['rescan_results'] = _rescan_result
                    except Exception as e:
                        st.error(f"❌ Tarama hatası: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # Show rescan results
        _rescan_df = st.session_state.get('rescan_results')
        if _rescan_df is not None and len(_rescan_df) > 0:
            st.markdown("---")
            st.markdown("### ✅ Yeniden Tarama Sonuçları")

            _chg_count = len(_rescan_df[_rescan_df['changed'] == '✅'])
            _same_count = len(_rescan_df) - _chg_count
            rr1, rr2, rr3 = st.columns(3)
            rr1.metric("Taranan Hücre", len(_rescan_df))
            rr2.metric("🔄 Değişen", _chg_count)
            rr3.metric("— Aynı", _same_count)

            st.dataframe(_rescan_df, use_container_width=True, height=400)

            _rescan_buf = io.BytesIO()
            with pd.ExcelWriter(_rescan_buf, engine='xlsxwriter') as w:
                _rescan_df.to_excel(w, sheet_name='Yeniden Tarama', index=False)
            st.download_button("📥 Tarama Sonuçlarını Excel İndir", _rescan_buf.getvalue(),
                               "pci_rsi_yeniden_tarama.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

# ============================================================
# TAB 6 — NEW CELL PCI/RSI FINDER
# ============================================================
with tab6:
    st.markdown("### 🆕 Yeni Hücre / Saha Ekleme — PCI ve RSI Bulma")
    st.markdown("""> Mevcut ağ verisini yükledikten sonra, **yeni eklenecek hücrelere** en uygun PCI ve RSI değerlerini bulur.
> Yeni hücrelerin komşulukları mevcut ağ üzerinden otomatik hesaplanır.""")

    if st.session_state.df is None:
        st.info("📌 Önce **Veri Yükleme** sekmesinden mevcut ağ verisini yükleyin.")
    else:
        df_existing = st.session_state.df

        st.markdown("---")
        input_method = st.radio("Yeni hücre bilgilerini nasıl girmek istiyorsunuz?",
                                ["📝 Manuel Giriş", "📤 Excel ile Yükleme"],
                                horizontal=True)

        new_cells_df = None

        if input_method == "📝 Manuel Giriş":
            st.markdown("#### 📝 Yeni Hücre Bilgileri")
            st.caption("Birden fazla hücre eklemek için tabloyu genişletin.")

            num_cells = st.number_input("Eklenecek hücre sayısı", 1, 50, 1)
            rows = []
            for i in range(num_cells):
                with st.expander(f"🔧 Hücre {i+1}", expanded=(i == 0)):
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        cid = st.text_input("Cell ID", key=f"nc_cid_{i}",
                                            placeholder="örn: TAS0500A")
                        site = st.text_input("Site ID", key=f"nc_site_{i}",
                                             placeholder="örn: AS0500")
                    with mc2:
                        lat = st.number_input("Enlem (Latitude)", -90.0, 90.0, 39.0,
                                              format="%.6f", key=f"nc_lat_{i}")
                        lon = st.number_input("Boylam (Longitude)", -180.0, 180.0, 32.0,
                                              format="%.6f", key=f"nc_lon_{i}")
                    with mc3:
                        az = st.number_input("Azimuth (°)", 0, 360, 0, key=f"nc_az_{i}")
                        bw = st.number_input("Beamwidth (°)", 10, 360, 65, key=f"nc_bw_{i}")

                    mc4, mc5 = st.columns(2)
                    with mc4:
                        zcz = st.number_input("Zero Correlation Zone", 0, 15, 5,
                                              key=f"nc_zcz_{i}")
                        prach_cfg = st.number_input("PRACH Config Index", 0, 63, 0,
                                                    key=f"nc_prach_{i}")
                    with mc5:
                        cell_range_m = st.number_input("Cell Range (m) — Huawei", 0, 120000, 0,
                                                       key=f"nc_cr_{i}",
                                                       help="Huawei cellRadius (metre). Girilirse ZCZ yerine bu değer kullanılır.")
                        earfcn = st.number_input("EARFCN/ARFCN", 0, 100000, 0,
                                                 key=f"nc_earfcn_{i}")

                    if cid:
                        row_data = {
                            'cell_id': cid, 'site_id': site,
                            'latitude': lat, 'longitude': lon,
                            'azimuth': az, 'beamwidth': bw,
                            'pci': None, 'rsi': None,
                            'zero_correlation_zone': zcz,
                            'prach_config_index': prach_cfg,
                            'earfcn': earfcn,
                        }
                        if cell_range_m > 0:
                            row_data['cell_range'] = cell_range_m
                        rows.append(row_data)

            if rows:
                new_cells_df = pd.DataFrame(rows)

        else:  # Excel upload
            st.markdown("#### 📤 Yeni Hücreleri Excel ile Yükle")
            st.markdown("""Excel dosyasında **en az** şu kolonlar olmalıdır:
- `cell_id`, `latitude`, `longitude`, `azimuth`

Opsiyonel: `site_id`, `beamwidth`, `zero_correlation_zone`, `prach_config_index`, `earfcn`, `cell_range` (Huawei cellRadius, metre)""")

            new_file = st.file_uploader("Yeni hücre Excel dosyası",
                                        type=['xlsx', 'xls'],
                                        key='new_cell_upload')
            if new_file is not None:
                try:
                    ndf = pd.read_excel(new_file)
                    # Normalize column names
                    ndf.columns = [c.strip().lower().replace(' ', '_') for c in ndf.columns]
                    # Check required columns
                    required = {'cell_id', 'latitude', 'longitude', 'azimuth'}
                    missing = required - set(ndf.columns)
                    if missing:
                        st.error(f"❌ Eksik kolonlar: {', '.join(missing)}")
                    else:
                        # Ensure optional columns exist
                        for col, default in [('site_id', ''), ('beamwidth', 65),
                                             ('pci', None), ('rsi', None),
                                             ('zero_correlation_zone', 5),
                                             ('prach_config_index', 0),
                                             ('earfcn', 0)]:
                            if col not in ndf.columns:
                                ndf[col] = default
                        new_cells_df = ndf
                        st.success(f"✅ {len(ndf)} yeni hücre yüklendi.")
                except Exception as e:
                    st.error(f"❌ Excel okuma hatası: {e}")

        if new_cells_df is not None and len(new_cells_df) > 0:
            st.markdown("---")
            st.markdown("#### 📋 Yeni Hücre Listesi")
            st.dataframe(new_cells_df[['cell_id', 'latitude', 'longitude', 'azimuth',
                                        'zero_correlation_zone', 'prach_config_index']],
                         use_container_width=True, hide_index=True)

            if st.button("🔍 Optimal PCI ve RSI Bul", type="primary", use_container_width=True):
                with st.spinner("Mevcut ağ komşulukları hesaplanıyor ve optimal değerler aranıyor..."):
                    try:
                        result = find_optimal_pci_rsi_for_new_cells(
                            existing_df=df_existing,
                            new_cells_df=new_cells_df,
                            radius_km=radius_km,
                            technology=tech,
                            use_antenna_direction=use_antenna,
                            default_beamwidth=float(default_bw),
                            check_mod3=check_mod3,
                            check_mod6=check_mod6,
                            check_mod30=check_mod30,
                            check_mod4=check_mod4,
                            sector_groups=st.session_state.sector_groups or {},
                            cell_to_sector=st.session_state.cell_to_sector or {},
                        )
                        st.session_state['new_cell_results'] = result
                    except Exception as e:
                        st.error(f"❌ Hata: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Show results
            nc_result = st.session_state.get('new_cell_results')
            if nc_result is not None and len(nc_result) > 0:
                st.markdown("---")
                st.markdown("### ✅ Sonuçlar — Önerilen PCI ve RSI Değerleri")

                ok_count = len(nc_result[nc_result['suggested_pci'] != '—'])
                fail_count = len(nc_result) - ok_count
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Toplam Hücre", len(nc_result))
                rc2.metric("✅ PCI Bulunan", ok_count)
                rc3.metric("❌ Bulunamayan", fail_count)

                st.dataframe(
                    nc_result,
                    use_container_width=True, height=400,
                    column_config={
                        'cell_id': st.column_config.TextColumn('Cell ID', width='medium'),
                        'suggested_pci': st.column_config.TextColumn('Önerilen PCI', width='small'),
                        'pss': st.column_config.TextColumn('PSS', width='small'),
                        'sss': st.column_config.TextColumn('SSS', width='small'),
                        'suggested_rsi': st.column_config.TextColumn('Önerilen RSI', width='small'),
                        'ncs': st.column_config.TextColumn('Ncs', width='small'),
                        'roots_needed': st.column_config.NumberColumn('Root Sayısı', width='small'),
                        'rsi_range': st.column_config.TextColumn('RSI Aralığı', width='small'),
                        'neighbors_found': st.column_config.NumberColumn('Komşu Sayısı', width='small'),
                        'pci_quality': st.column_config.TextColumn('PCI Kalitesi', width='medium'),
                        'reason': st.column_config.TextColumn('Açıklama', width='large'),
                    })

                # Detail expander
                with st.expander("📋 Hücre Bazında Detaylar"):
                    for _, row in nc_result.iterrows():
                        cid = row['cell_id']
                        pci = row['suggested_pci']
                        rsi = row['suggested_rsi']
                        st.markdown(f"""**{cid}**:
- PCI: **{pci}** (PSS={row['pss']}, SSS={row['sss']}) — {row['pci_quality']}
- RSI: **{rsi}** (Ncs={row['ncs']}, {row['roots_needed']} root, aralık: {row['rsi_range']})
- Bulunan komşu sayısı: {row['neighbors_found']}
- {row['reason']}""")
                        st.markdown("---")

                # Download
                nc_buf = io.BytesIO()
                with pd.ExcelWriter(nc_buf, engine='xlsxwriter') as w:
                    nc_result.to_excel(w, sheet_name='Yeni Hücre PCI-RSI', index=False)
                st.download_button("📥 Sonuçları Excel Olarak İndir", nc_buf.getvalue(),
                                   "yeni_hucre_pci_rsi.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

# ============================================================
# TAB 7 — INFO
# ============================================================
with tab7:
    st.markdown("### ℹ️ TürkTelekom PCI/RSI Planner v2.0")

    info_section = st.selectbox("Bilgi Bölümü Seçin", [
        "📡 PCI Nedir? — Temel Kavramlar",
        "🔍 PCI Analizi Nasıl Çalışır? — Çakışma Tespiti",
        "🚀 PCI Otomatik Planlayıcı — Sıfırdan Atama (SA)",
        "💡 PCI/RSI Öneriler & Tarama — Greedy",
        "📻 RSI Nedir? — PRACH ve Root Sequence",
        "🔬 RSI Analizi Nasıl Çalışır? — Cell Range",
        "📻 RSI Otomatik Planlayıcı — Greedy Atama",
        "📶 Band Tanıma ve Sektör Grupları",
        "🗺️ Harita Gösterimi",
    ])

    st.markdown("---")

    # ========================================
    if "PCI Nedir" in info_section:
        st.markdown("""
## 📡 Physical Cell Identity (PCI) — Temel Kavramlar

### PCI Nedir?
PCI (Physical Cell Identity), LTE ve NR ağlarında her hücreye atanan **fiziksel katman tanımlayıcısıdır**.
UE (kullanıcı cihazı) bir hücreyi ilk tespit ettiğinde, o hücrenin PCI'sini okuyarak senkronizasyon sağlar.

### PCI Yapısı
PCI iki bileşenin kombinasyonudur:

```
PCI = 3 × SSS + PSS
```

| Bileşen | Tam Adı | Değer Aralığı | Görevi |
|---|---|---|---|
| **PSS** | Primary Synchronization Signal | 0, 1, 2 (3 değer) | İlk zaman senkronizasyonu, 5ms yarı-frame sınır tespiti |
| **SSS** | Secondary Synchronization Signal | LTE: 0-167 / NR: 0-335 | Frame sınırı, Cell ID Group tespiti |

### PCI Sayıları
| | LTE (4G) | NR (5G) |
|---|---|---|
| **Toplam PCI** | 504 (0-503) | 1008 (0-1007) |
| **PSS Sayısı** | 3 | 3 |
| **SSS Sayısı** | 168 | 336 |

### Neden Önemli?
- PCI **sınırlı bir kaynaktır** — tüm ağda sadece 504 (LTE) veya 1008 (NR) değer mevcuttur
- Komşu hücreler arası **çakışma olmamalıdır** — yoksa UE yanlış hücreye bağlanır
- PCI'nin mod3 değeri (=PSS) frekans referans sinyallerinin mapping'ini belirler
- Yanlış PCI ataması → **düşük hücre çıkışı, handover hataları, throughput kaybı**
""")

    # ========================================
    elif "PCI Analizi" in info_section:
        st.markdown("""
## 🔍 PCI Analizi Nasıl Çalışır? — Çakışma Tespit Algoritması

Sistem yüklenen hücre verisini **4 aşamada** analiz eder:

---

### Aşama 1: Komşuluk Keşfi
Hangi hücrelerin birbirini etkilediğini bulmak için **3 yöntem** birlikte kullanılır:

**1️⃣ Mesafe + Anten Yönü (Otomatik)**
```
Her hücre çifti (i, j) için:
  1. Haversine mesafe = d(lat_i, lon_i, lat_j, lon_j)
  2. Eğer d ≤ tarama_yarıçapı:
     3. Bearing (yön açısı) hesapla: i→j ve j→i
     4. Eğer bearing, i'nin anten hüzmesi içindeyse VEYA
        bearing, j'nin anten hüzmesi içindeyse:
        → i ve j KOMŞU!
```
> **Anten hüzmesi kontrolü:** `|bearing - azimuth| ≤ beamwidth/2` (±180° wrap ile)

**2️⃣ Aynı Site Sektörleri (İç Komşuluk)**
```
Aynı site_id'ye sahip tüm hücreler → otomatik KOMŞU
```

**3️⃣ Harici Komşuluk Listesi (Opsiyonel)**
```
Excel'den yüklenen cell_1 ↔ cell_2 çiftleri → doğrudan KOMŞU olarak eklenir
```

---

### Aşama 2: Çakışma Tespiti

Her komşu çifti, **teknolojiye göre uygun kontroller** ile test edilir:

#### 🔴 Collision (KRİTİK) — LTE + NR
```
A.pci == B.pci  VE  A ↔ B komşu
→ UE her iki hücreden aynı PSS+SSS alır → senkronizasyon başarısız!
```

#### 🟠 Confusion (YÜKSEK) — LTE + NR
```
B.pci == C.pci  VE  hem B hem C, A'nın komşusu
→ A, handover hedefi olarak B/C'yi ayırt edemez!
```

#### 🟡 Mod 3 Conflict (ORTA) — LTE + NR
```
A.pci % 3 == B.pci % 3  →  Aynı PSS → interferans
```

#### 🟤 Mod 4 Conflict (ORTA) — Yalnızca NR
```
A.pci % 4 == B.pci % 4  →  SSB DMRS sekansı çakışır
```
> 3GPP TS 38.211: NR'da SSB DMRS sekansı `PCI mod 4` ile belirlenir.
> Bu kontrol otomatik olarak NR seçildiğinde aktif, LTE'de devre dışıdır.

#### 🔵 Mod 6 Conflict (DÜŞÜK) — Yalnızca LTE
```
A.pci % 6 == B.pci % 6  →  Reference Signal (RS) frekans kaydırma çakışması
```

#### ⚪ Mod 30 Conflict (DÜŞÜK) — Yalnızca LTE
```
A.pci % 30 == B.pci % 30  →  PCFICH / PHICH kaynak çakışması
```

> **Teknolojiye Göre Otomatik Geçiş:** NR seçildiğinde Mod4 aktif, Mod6/Mod30 devre dışı.
> LTE seçildiğinde Mod6/Mod30 aktif, Mod4 devre dışı. Mod3 her ikisinde de aktif.

---

### Aşama 3: Sağlık Skoru Hesaplama

Skor, **teknolojiye göre farklı ağırlıklar** kullanır:

**LTE Ağırlıkları:**
| Çakışma | Ağırlık (W) | Fonksiyon | Açıklama |
|---|:---:|---|---|
| Collision | 15 | √(oran) | Kritik — kök fonksiyon (küçük sayıda bile cezalı) |
| Confusion | 20 | lineer | Yüksek — doğrusal ceza |
| RSI | 20 | √(oran) | Yüksek — kök fonksiyon |
| Mod 3 | 20 | excess>50% | %50 eşik, aşan kısım cezalanır |
| Mod 6 | 15 | excess>60% | %60 eşik, aşan kısım cezalanır |
| Mod 30 | 10 | excess>70% | %70 eşik, aşan kısım cezalanır |

**NR Ağırlıkları:**
| Çakışma | Ağırlık (W) | Fonksiyon | Açıklama |
|---|:---:|---|---|
| Collision | 15 | √(oran) | Kritik |
| Confusion | 20 | lineer | Yüksek |
| RSI | 20 | √(oran) | Yüksek |
| Mod 3 | 25 | excess>50% | PSS (Mod6/30 yok, ağırlık artar) |
| Mod 4 | 20 | excess>50% | SSB DMRS (NR'ye özel) |

```
Skor = 100 - Σ(W_i × penalty_i)    →  [0, 100]
```
> 80+ = İyi 🟢, 50-80 = Orta 🟡, <50 = Kötü 🔴
""")

    # ========================================
    elif "PCI Otomatik" in info_section:
        st.markdown("""
## 🚀 PCI Otomatik Planlayıcı — Simulated Annealing (SA)

Öneri algoritmasından farklı olarak, **tüm hücrelere sıfırdan** PCI atar.
**Simulated Annealing (SA)** metaheuristic optimizasyon algoritması kullanılır.

---

### SA (Simulated Annealing) Nedir?
Metallerin yavaş soğutma ile kristal yapıya ulaşması sürecinden esinlenmiştir.
- **Başlangıç sıcaklığı (T₀=50):** Yüksek sıcaklıkta kötü çözümler de kabul edilir (lokal minimumdan kaçış)
- **Soğuma:** T geometrik olarak azalır → giderek daha seçici
- **Bitiş sıcaklığı (T_end=0.01):** Neredeyse sadece iyileştirme kabul edilir
- **İterasyon sayısı:** min(max(300K, n_cells×400), 800K)

### Enerji Fonksiyonu (Teknolojiye Göre)
Her hücrenin "enerji" değeri hesaplanır. Düşük enerji = iyi PCI.

| Çakışma Türü | Ağırlık | LTE | NR | Açıklama |
|---|:---:|:---:|:---:|---|
| **Co-site Collision** | ∞ (hard) | ✅ | ✅ | Aynı sitede aynı PCI → yasaklı set ile engellenir |
| **Co-site Mod3 (outdoor↔outdoor)** | 200 | ✅ | ✅ | Aynı sitede aynı PSS (outdoor hücreler arası) |
| **Co-site Mod3 (indoor)** | 80 | ✅ | ✅ | Aynı sitede aynı PSS (en az bir indoor hücre) |
| **Collision** | 100 | ✅ | ✅ | Komşu hücrelerde aynı PCI |
| **Confusion** | 30 | ✅ | ✅ | 2-hop çakışma (handover belirsizliği) |
| **Mod3** | 8 | ✅ | ✅ | Aynı PSS → referans sinyal interferansı |
| **Mod4 (SSB DMRS)** | 2.5 | ❌ | ✅ | NR SSB DMRS sekansı çakışması |
| **Mod6 (RS)** | 1.5 | ✅ | ❌ | LTE referans sinyal frekans kaydırma |
| **Mod30 (PCFICH)** | 0.5 | ✅ | ❌ | LTE PCFICH/PHICH kaynak çakışması |

> **Co-site Collision = Hard Constraint:** Aynı fiziksel sitedeki farklı sektörler
> asla aynı PCI'yi kullanamazlar. SA bu kısıtı `_co_site_forbidden()` fonksiyonu ile
> yasaklı PCI seti olarak uygular — ağırlık değil, kesin engel.

```
E(cell) = W_COL × collision_count
        + W_COSITE_M3 × co_site_mod3_outdoor_count
        + W_COSITE_M3_INDOOR × co_site_mod3_indoor_count
        + W_CON × confusion_count
        + W_MOD3 × mod3_count
        + W_MOD4 × mod4_count     (NR only)
        + W_MOD6 × mod6_count     (LTE only)
        + W_MOD30 × mod30_count   (LTE only)
  + co_site_collision → YASAK (kesin engel)
```

### SA Döngüsü
```
1. Rastgele bir hücre seç
2. Mevcut enerjiyi hesapla → E_old
3. Yeni rastgele PCI ata (mod3 tercihi ile)
4. Yeni enerjiyi hesapla → E_new
5. ΔE = E_new - E_old
   - ΔE < 0 → İyileşme, HER ZAMAN kabul et
   - ΔE ≥ 0 → Kötüleşme, exp(-ΔE/T) olasılıkla kabul et
6. Sıcaklığı düşür: T = T × cooling_rate
7. Sektör propagasyonu: değişen PCI → aynı sektördeki tüm hücrelere kopyalanır
```

### Çok Aşamalı Gevşetme (Post-SA Fixup)
SA sonrası hâlâ atanamayan hücreler için **kademeli gevşetme** uygulanır:

**LTE Gevşetme Seviyeleri:**
```
Pass 1 — TAM UYUM: collision + mod3 + mod6 + mod30 + confusion + co-site mod3
Pass 2 — MOD30 GEVŞETİLDİ
Pass 3 — MOD6 GEVŞETİLDİ
Pass 4 — MOD3 GEVŞETİLDİ
Pass 5 — CONFUSION GEVŞETİLDİ
Pass 6 — CO-SITE MOD3 GEVŞETİLDİ
Pass 7 — SADECE COLLISION (son çare)
```

**NR Gevşetme Seviyeleri:**
```
Pass 1 — TAM UYUM: collision + mod3 + mod4 + confusion + co-site mod3
Pass 2 — MOD4 GEVŞETİLDİ
Pass 3 — MOD3 GEVŞETİLDİ
Pass 4 — CONFUSION GEVŞETİLDİ
Pass 5 — CO-SITE MOD3 GEVŞETİLDİ
Pass 6 — SADECE COLLISION (son çare)
```

### Sonuç Tablosu
- **current_pci** → **planned_pci**: Eski ve yeni PCI
- **changed**: Değişti mi?
- **relaxation_level**: Hangi seviyede atandı?
""")

    # ========================================
    elif "Öneriler" in info_section:
        st.markdown("""
## 💡 PCI/RSI Öneriler & Tarama — Greedy Algoritma

Bu bölüm, **mevcut ağ üzerinde en az değişiklik** prensibiyle çalışır.
SA'dan farklı olarak sıfırdan atama yapmaz — sadece sorunlu hücrelere yeni değer önerir.

---

### PCI Önerisi Nasıl Çalışır?

**1. Sorunlu Hücre Tespiti:**
Sadece **Collision, Confusion, Co-site Collision veya Co-site Mod3** problemi olan
hücreler hedeflenir. Yalnızca ModN çakışması olan hücreler değiştirilmez (kozmetik).

**2. Önceliklendirme:**
```
Öncelik sırası (yüksek → düşük):
  1. Co-site Collision (aynı sitede aynı PCI)
  2. Collision (komşuda aynı PCI)
  3. Confusion + Co-site Mod3
  4. Mod3, Mod4 (NR)
  5. Mod6, Mod30 (LTE)
```

**3. Aday PCI Arama (Spiral Search):**
```
Mevcut PCI'den başlayarak spiral düzende arama:
  → En yakın PCI'ler önce denenir (minimum değişiklik)
  → Tercih edilen mod3 sınıfı öncelikli (komşularda az kullanılan)
  → Co-site PCI'ler yasaklı setten çıkarılır (hard constraint)
```

**4. Kademeli Gevşetme (LTE):**
```
Seviye 1 — TAM UYUM: mod3 + mod6 + mod30 + confusion + co-site mod3
Seviye 2 — MOD30 gevşetildi
Seviye 3 — MOD6 gevşetildi
Seviye 4 — MOD3 gevşetildi
Seviye 5 — CONFUSION gevşetildi
Seviye 6 — CO-SITE MOD3 gevşetildi
Seviye 7 — SADECE COLLISION (son çare)
```

**NR'da:** Mod6/Mod30 otomatik devre dışı, Mod4 aktif.

**5. Güvenlik Kontrolü (_quick_score):**
Aday PCI bulunduğunda, SA ile **birebir aynı ağırlıklar** kullanılarak
eski ve yeni skor karşılaştırılır. Yeni skor daha kötüyse → öneri reddedilir.

| Çakışma Türü | Ağırlık | LTE | NR |
|---|:---:|:---:|:---:|
| **Co-site Collision** | 500 (hard) | ✅ | ✅ |
| **Co-site Mod3 (outdoor)** | 200 | ✅ | ✅ |
| **Co-site Mod3 (indoor)** | 80 | ✅ | ✅ |
| **Collision** | 100 | ✅ | ✅ |
| **Confusion** | 30 | ✅ | ✅ |
| **Mod3** | 8 | ✅ | ✅ |
| **Mod4** | 2.5 | ❌ | ✅ |
| **Mod6** | 1.5 | ✅ | ❌ |
| **Mod30** | 0.5 | ✅ | ❌ |

> Bu ağırlıklar SA enerji fonksiyonu ile eşleştirilmiştir.
> Co-site Collision, SA'da hard constraint (yasaklı set) olduğu için
> burada çok yüksek bir ceza (500) ile simüle edilir.

**6. Sektör Propagasyonu:**
Bir sektördeki lider hücreye PCI atandığında, **aynı sektördeki tüm co-sector hücreler**
otomatik olarak aynı PCI'yi alır ve tabloda kendi satırlarıyla listelenir.

---

### RSI Önerisi Nasıl Çalışır?
- Sadece **RSI çakışması** olan hücreler hedeflenir
- Cell-range-aware kontrol: root aralığı örtüşmesi aranır
- Co-sector hücreler aynı RSI'yi alır
- Lider hücreye atanan RSI, sektördeki tüm hücrelere propagate edilir

---

### Seçici Yeniden Tarama (Rescan)
Tüm ağ yerine **sadece seçilen hücreleri** yeniden tarar.
- Site bazlı veya cell bazlı seçim yapılabilir
- PCI ve/veya RSI ayrı ayrı taranabilir
- Aynı gevşetme ve co-site kuralları geçerlidir
- Tarama sonucu: mevcut PCI/RSI korunur veya daha iyi değer önerilir
""")

    # ========================================
    elif "RSI Nedir" in info_section:
        st.markdown("""
## 📻 RSI Nedir? — PRACH ve Root Sequence Kavramları

### RSI (Root Sequence Index) Nedir?
RSI, hücrelerin **PRACH (Physical Random Access Channel)** kanalında kullandığı
Zadoff-Chu root sequence'ın başlangıç indeksidir.

UE bir hücreye **Random Access** yapmak istediğinde (ilk bağlantı, handover, vb.)
PRACH üzerinden bir **preamble** gönderir. Bu preamble, hücrenin RSI'sinden türetilir.

### LTE vs NR Karşılaştırması

| Özellik | LTE (4G) | NR (5G) Long | NR (5G) Short |
|---|---|---|---|
| **Nzc** | 839 | 839 | 139 |
| **RSI Aralığı** | 0–837 | 0–837 | 0–137 |
| **PRACH Config** | 0–63 | 0–27 | ≥28 (Format A1-C2) |
| **Tseq** | 800 / 1600 µs | 800 / 1600 µs | 133.33 µs |
| **Kullanım** | Tüm LTE | Makro hücreler | Küçük hücreler, FR2 |

### NR Preamble Format Tespiti
Sistem, **prach_config_index** değerine göre otomatik algılar:
- **0–27** → Long sequence (L=839, Tseq=800-1600 µs)
- **≥28** → Short sequence (L=139, Tseq=133.33 µs)

> NR'da **her hücre farklı format** kullanabilir! Sistem hücre bazında `max_rsi`
> değerini otomatik belirler (L=839 → max 837, L=139 → max 137).

### Ncs (Cyclic Shift)
**zeroCorrelationZoneConfig** parametresi → **Ncs** değeri belirler.
LTE ve NR **farklı Ncs tabloları** kullanır (3GPP TS 36.211 / 38.211).

```
Preambles per Root = floor(Nzc / Ncs)
Roots Needed = ceil(64 / Preambles per Root)
```

### LTE Ncs Tablosu (Unrestricted, Nzc=839)
| Config | Ncs | Preamble/Root | Roots | Cell Range (km) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 13 | 64 | 1 | 1.86 |
| 5 | 26 | 32 | 2 | 3.72 |
| 8 | 46 | 18 | 4 | 6.58 |
| 11 | 93 | 9 | 8 | 13.30 |
| 12 | 119 | 7 | 10 | 17.02 |
| 15 | 419 | 2 | 32 | 59.93 |

### Örnek
**Ncs=26 → 32 preamble/root → 2 root gerekli**
RSI=42 olan hücre → root **42** ve **43** kullanır

**NR Short (Ncs=2, L=139) → 69 preamble/root → 1 root gerekli**
RSI=10 olan hücre → sadece root **10** kullanır

### 🏭 Vendor Desteği — Huawei cellRange Modu
**Nokia**: Excel'de `prachConfigIndex` ve `zeroCorrelationZoneConfig` sütunları doğrudan bulunur.

**Huawei**: Bu parametreler yerine **cellRange** (hücre yarıçapı, metre) bilgisi verilir.
Sistem, cellRange sütunu tespit ettiğinde otomatik olarak **Huawei modu**na geçer:

```
cellRange (m) → km'ye çevir
→ Gereken minimum Ncs hesapla: Ncs = range_km × Nzc × 2000 / (Tseq × c)
→ Ncs tablosundan en yakın (≥) zcz config değerini bul
→ RSI planlaması bu zcz ile yapılır
```

| cellRange (m) | Hesaplanan Ncs | Eşleşen zcz | Gerçek Kapsama |
|:---:|:---:|:---:|:---:|
| 3000 | ≥21.0 | 4 (Ncs=22) | 3.15 km |
| 14500 | ≥101.4 | 12 (Ncs=119) | 17.02 km |
| 29500 | ≥206.3 | 14 (Ncs=279) | 39.89 km |
| 38000 | ≥265.7 | 14 (Ncs=279) | 39.89 km |

> **Öncelik:** Bir hücrede hem `cell_range` hem `zero_correlation_zone` varsa, `cell_range` kullanılır.
""")

    # ========================================
    elif "RSI Analizi" in info_section:
        st.markdown("""
## 🔬 RSI Analizi Nasıl Çalışır? — Cell Range Tabanlı Çakışma Tespiti

### Neden "Basit RSI Eşitlik Kontrolü" Yetmez?
Klasik yaklaşım: "Komşu hücrelerin RSI'si aynı mı?" → Bu **YETERSİZ**!

Çünkü bir hücre RSI'sinden başlayarak **birden fazla ardışık root sequence** kullanır.
Bu aralıklar örtüşürse → preamble çakışması olur.

### Cell-Range-Aware RSI Çakışma Tespiti

```
Her komşu çifti (A, B) için:
  1. A'nın Ncs, Nzc, roots_needed değerlerini hesapla
  2. B'nin Ncs, Nzc, roots_needed değerlerini hesapla
  3. pair_max = min(A.max_rsi, B.max_rsi)   ← NR'da önemli
  4. A'nın kullandığı root aralığı:
     {RSI_A mod pair_max, ..., (RSI_A + roots_A - 1) mod pair_max}
  5. B'nin kullandığı root aralığı:
     {RSI_B mod pair_max, ..., (RSI_B + roots_B - 1) mod pair_max}
  6. Eğer bu iki küme kesişiyorsa → RSI ÇAKIŞMASI!
```

### NR'da Per-Cell Max RSI
NR'da her hücre **farklı preamble formatı** kullanabilir:

| PRACH Config | Sequence | Nzc | Max RSI | Wrapping |
|:---:|:---:|:---:|:---:|:---:|
| 0–27 | Long | 839 | **837** | mod 838 |
| ≥ 28 | Short | 139 | **137** | mod 138 |

Karışık formatlı komşu çifti kontrolünde:
```
pair_max_rsi = min(cell_A.max_rsi, cell_B.max_rsi)
```
Bu sayede L=839 ve L=139 hücreleri aynı ağda güvenle analiz edilir.

### Görsel Örnekler

**LTE / NR Long (Nzc=839):**
```
Hücre A: RSI=40, Ncs=26 → 2 root → aralık [40, 41]
Hücre B: RSI=41, Ncs=26 → 2 root → aralık [41, 42]
                                         ^^
                                    Root 41 ÇAKIŞIYOR!
```

**Farklı Ncs Değerleri:**
```
Hücre X: RSI=100, Ncs=119 → 10 root → aralık [100-109]
Hücre Y: RSI=108, Ncs=26  →  2 root → aralık [108-109]
                                         ^^^^^^^^
                                    Root 108,109 ÇAKIŞIYOR!
```

**NR Short Sequence (Nzc=139):**
```
Hücre P: RSI=130, pcfg=30 (short) → max_rsi=137
  Ncs=2 → 1 root → aralık [130]
Hücre Q: RSI=135, pcfg=30 (short) → max_rsi=137
  Ncs=2 → 1 root → aralık [135]
  Çakışma YOK ✅ (farklı root indeksleri)
```

### Cell Range (Hücre Menzili) Hesabı
```
Cell Range (km) = (Ncs / Nzc) × Tseq × c / 2
```

| Parametre | LTE (Format 0-3) | NR Long | NR Short |
|:---:|:---:|:---:|:---:|
| **Nzc** | 839 | 839 | 139 |
| **Tseq** | 800 / 1600 µs | 800 µs | 133.33 µs |
| **c** | 3×10⁸ m/s | 3×10⁸ m/s | 3×10⁸ m/s |

- Bu mesafe, hücrenin PRACH'ı doğru algılayabileceği maksimum mesafedir.
""")

    # ========================================
    elif "RSI Otomatik" in info_section:
        st.markdown("""
## 📻 RSI Otomatik Planlayıcı — Greedy Interval Graph Coloring

Tüm ağ için sıfırdan RSI atar — **Greedy Interval Graph Coloring**:

```
1. Ön hazırlık:
   a. Her hücrenin prach_config_index'inden format tespit et
      ─ NR: get_nr_preamble_info(pcfg) → (is_short, nzc, tseq)
      ─ LTE: Nzc=839, max_rsi=838 (sabit)
   b. Her hücrenin roots_needed değerini hesapla
   c. cell_max_rsi sözlüğü oluştur (hücre bazında wrapping sınırı)

2. Hücreleri sırala (ZORLUK SIRALAMASI):
   → Önce en çok root ihtiyacı olan (geniş aralık kaplayanlar)
   → Eşitlikte en çok komşusu olan (en kısıtlı hücreler)

3. Her hücre için:
   a. 1. ve 2. halka komşularının kullandığı root indekslerini topla
      → occupied = {komşunun tüm root indeksleri mod pair_max}
   b. RSI=0'dan cell_max_rsi'ye kadar tara
   c. [RSI, RSI + roots_needed) aralığı (mod cell_max_rsi)
      hiçbir occupied root ile örtüşmüyorsa → ATA
   d. Çalışan haritayı güncelle

4. Wrapping: RSI, hücrenin max_rsi'sine ulaşınca 0'a sarar
   ─ LTE: mod 838    (RSI 0-837 arası döngü)
   ─ NR Long: mod 838
   ─ NR Short: mod 138  (RSI 0-137 arası döngü)
```

### NR Format Tespiti
```python
# get_nr_preamble_info(prach_config_index) fonksiyonu:
#   pcfg 0-27  → (False, 839, tseq_us)  # Long sequence
#   pcfg 28+   → (True,  139, tseq_us)  # Short sequence
```
Bu tespit **her hücre için ayrı ayrı** yapılır, böylece aynı ağda
hem L=839 hem L=139 hücreler bulunabilir.

### Neden En Zor Hücre Önce?
Çok root kullanan hücreler (büyük Ncs) daha geniş RSI aralığı kaplar.
Bunlara önce atama yapmak, dar aralıklı hücrelere daha çok boşluk bırakır.

Örnek sıralama:
```
1. Ncs=119, 10 root gerekli → ÖNCELİKLİ (geniş aralık)
2. Ncs=46,   4 root gerekli → İKİNCİ
3. Ncs=26,   2 root gerekli → SON (dar aralık, her yere sığar)
```
""")

    # ========================================
    elif "Band Tanıma" in info_section:
        st.markdown("""
## 📶 Band Tanıma ve Sektör Grupları

### Hücre İsimlendirme Kuralı
Hücre ID'sinin **ilk harfi** ve **son harfi** bandı ve bant genişliğini belirler:

| İlk Harf | Son Harf | Band | Bant Genişliği | Açıklama |
|:---:|:---:|:---:|:---:|---|
| **Z** | — | 2100 MHz | — | UMTS/LTE 2100 bandı |
| **E** | — | 2600 MHz | — | LTE 2600 bandı (yüksek kapasite) |
| **L** | — | 800 MHz | — | LTE 800 bandı (geniş kapsama) |
| **T** | A, D, G, J, M, P | 1800 MHz | 20 MHz | LTE 1800 geniş bant |
| **T** | B, E, H, K, N, R | 1800 MHz | 10 MHz | LTE 1800 dar bant |

### Sektör Grubu Nedir?
Bir baz istasyonunda (site) **aynı yöne bakan** (azimuth ±10°) hücreler aynı **fiziksel sektördedir**.

```
Örnek: SITE001 sahasında 3 sektör, her sektörde 3 band:

Sektör 1 (Az=0°):   L_SITE001_A (800)  + T_SITE001_A (1800/20) + Z_SITE001_A (2100)
Sektör 2 (Az=120°):  L_SITE001_B (800)  + T_SITE001_B (1800/20) + Z_SITE001_B (2100)
Sektör 3 (Az=240°):  L_SITE001_C (800)  + T_SITE001_C (1800/20) + Z_SITE001_C (2100)
```

### Aynı Sektörde Aynı PCI Kuralı
**Aynı sektördeki tüm hücreler aynı PCI'yi KULLANMALIDIR.**

Neden?
- Aynı fiziksel antenlerden yayın yaparlar
- UE bunları tek bir "mantıksal hücre" gibi görür (carrier aggregation)
- Farklı PCI atanırsa → UE gereksiz ölçüm ve raporlama yapar
- Handover algoritmaları karışır

Sistem bu kuralı **otomatik olarak uygular**:
- **`sector` kolonu doluysa**: Doğrudan bu kolon referans alınır (naming convention ve azimuth devre dışı)
- **`sector` kolonu yoksa veya boşsa**: Sektör grupları azimuth ±10° toleransla ve naming convention ile tespit edilir
- PCI önerisi veya planlaması yapılırken, bir sektördeki herhangi bir hücreye PCI atandığında
  aynı sektördeki tüm hücrelere otomatik olarak aynı PCI kopyalanır
""")

    # ========================================
    elif "Harita" in info_section:
        st.markdown("""
## 🗺️ Harita Gösterimi

### Katmanlı Band Görünümü
Haritada her frekans bandı **farklı renk** ve **farklı boyutta** sektör dilimleri ile gösterilir:

| Band | Renk | Dilim Boyutu | Neden? |
|---|---|---|---|
| 🟢 **800 MHz** | Yeşil | En büyük (×1.6) | En geniş kapsama alanı (düşük frekans) |
| 🟣 **1800 MHz** | Mor | Orta (×1.0) | Orta kapsama |
| 🔵 **1800/20 MHz** | Mavi | Orta (×1.0) | 20 MHz bant genişliği, yüksek kapasite |
| 🟠 **1800/10 MHz** | Turuncu | Küçük (×0.9) | 10 MHz bant genişliği |
| 🔴 **2100 MHz** | Kırmızı | Küçük (×0.75) | Daha kısa menzil |
| 🔵 **2600 MHz** | Teal | En küçük (×0.55) | En kısa menzil (yüksek frekans) |

### İnteraktif Özellikler
- **Katman Kontrolü**: Haritanın sağ üstündeki panel ile her bandı ayrı ayrı açıp kapatabilirsiniz
- **Sektör Tıklama**: Bir sektör dilimine tıklayın → o hücrenin tüm detayları (PCI, RSI, band, azimuth, sektör grubu)
- **Site İşaretçisi**: Kule ikonuna tıklayın → o sitedeki TÜM hücrelerin özet tablosu
- **Çakışma Vurgusu**: Çakışması olan hücreler kırmızı ile vurgulanır

### Sektör Dilimleri Neyi Temsil Ediyor?
- Dilimin **yönü** = antenin azimuth açısı
- Dilimin **genişliği** = anten hüzme genişliği (beamwidth)
- Dilimin **uzunluğu** = banda göre göreceli kapsama alanı
- Dilimin **rengi** = frekans bandı

> Not: Dilim uzunlukları gerçek kapsama mesafesini değil, **göreceli farkı** gösterir.
> Gerçek propagasyon birçok faktöre bağlıdır (anten kazancı, güç, arazi, vb.)
""")

    st.markdown("---")
    st.caption("3GPP TS 36.211 v17.0.0 / 3GPP TS 38.211 v17.0.0 • TürkTelekom PCI/RSI Planner v2.0")
