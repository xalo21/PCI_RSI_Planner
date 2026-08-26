"""
Excel Import/Export Module for PCI/RSI Planner v2.0
====================================================
Now includes PRACH Config Index and zeroCorrelationZoneConfig columns.
"""

import pandas as pd
import numpy as np
import io
import re
from typing import Tuple, Optional


def _parse_concatenated_dms(val):
    """Parse concatenated DMS integer like 38222968 -> 38°22'29.68".

    Format: DDMMSSss  (or DDDMMSSss for 3-digit degrees like longitude)
    - Last 4 digits: seconds × 100  (2968 → 29.68")
    - Next 2 digits: minutes         (22 → 22')
    - Remaining: degrees              (38 → 38°)

    Returns decimal degrees as float.
    """
    s = str(abs(int(val)))
    if len(s) < 7:
        return np.nan
    sec_part = s[-4:]          # SSss
    secs = int(sec_part[:2]) + int(sec_part[2:]) / 100.0
    min_part = s[-6:-4]        # MM
    mins = int(min_part)
    deg_part = s[:-6]          # DD or DDD
    degs = int(deg_part)
    dd = degs + mins / 60.0 + secs / 3600.0
    if val < 0:
        dd = -dd
    return dd


def _parse_coordinate(val):
    """Parse a coordinate value from various formats to decimal degrees.

    Supported formats:
      - Decimal degrees (float): 38.222968
      - Decimal degrees with comma: '38,222968'
      - Concatenated DMS integer: 38222968 → 38°22'29.68"
      - DMS: '38° 13\' 22.68\"', '38 13 22.68', '38°13'22.68"'
      - Degrees + decimal minutes: '38 13.378'
    Returns float or NaN.
    """
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return np.nan
        # Detect concatenated DMS: integer value > 1_000_000
        if isinstance(val, int) or (isinstance(val, float) and val == int(val)):
            ival = int(val)
            if abs(ival) > 1_000_000:
                return _parse_concatenated_dms(ival)
        return float(val)

    s = str(val).strip()
    if not s:
        return np.nan

    # Replace comma with dot for Turkish/European decimal separator
    # But only if there's exactly one comma and no dot (to avoid misinterpreting)
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')

    # Try direct float parse first (handles '38.222968' and '38,222968' after replace)
    try:
        return float(s)
    except ValueError:
        pass

    # Try DMS pattern: 38° 13' 22.68" or 38 13 22.68 or 38°13'22.68"
    # Also handles negative / S / W
    dms_pattern = r'([+-]?)\s*(\d+)[°\s]+(\d+)[\'′\s]+(\d+(?:[.,]\d+)?)[\"″\s]*([NSEW]?)'
    m = re.match(dms_pattern, s, re.IGNORECASE)
    if m:
        sign_prefix = m.group(1)
        deg = float(m.group(2))
        mins = float(m.group(3))
        secs = float(m.group(4).replace(',', '.'))
        direction = m.group(5).upper()
        dd = deg + mins / 60.0 + secs / 3600.0
        if sign_prefix == '-' or direction in ('S', 'W'):
            dd = -dd
        return dd

    # Try degrees + decimal minutes: 38 13.378
    dm_pattern = r'([+-]?)\s*(\d+)[°\s]+(\d+(?:[.,]\d+)?)[\'′\s]*([NSEW]?)'
    m = re.match(dm_pattern, s, re.IGNORECASE)
    if m:
        sign_prefix = m.group(1)
        deg = float(m.group(2))
        mins = float(m.group(3).replace(',', '.'))
        direction = m.group(4).upper()
        dd = deg + mins / 60.0
        if sign_prefix == '-' or direction in ('S', 'W'):
            dd = -dd
        return dd

    return np.nan


def _clean_coordinate_column(series):
    """Apply coordinate parsing to an entire pandas Series.
    Returns cleaned float Series with any unparseable values as NaN."""
    return series.apply(_parse_coordinate).astype(float)

COLUMN_ALIASES = {
    'cell_id': ['cell_id','cellid','cell_name','cellname','ci','cell','hücre','hucre','site_cell'],
    'site_id': ['site_id','siteid','site_name','sitename','site','saha'],
    'latitude': ['latitude','lat','enlem','y'],
    'longitude': ['longitude','lon','lng','long','boylam','x'],
    'azimuth': ['azimuth','azimut','antenna_azimuth','anten_yonu','yon','direction','bearing'],
    'pci': ['pci','physical_cell_id','physicalcellid','nrpci','phy_cell_id'],
    'rsi': ['rsi','root_sequence_index','rootsequenceindex','prach_rsi','prachrootsequenceindex'],
    'beamwidth': ['beamwidth','beam_width','hbw','horizontal_beamwidth','huzme'],
    'technology': ['technology','tech','rat','teknoloji','network_type'],
    'band': ['band','frequency_band','frekans','band_mhz','freq_band'],
    # Explicit carrier frequency.  This is the only reliable carrier key: a
    # band alone cannot separate two carriers within the same band, and the
    # cell-ID prefix cannot either.  Kept separate from 'band' on purpose.
    'earfcn': ['earfcn','arfcn','nrarfcn','nr_arfcn','dl_earfcn','earfcn_dl',
               'ssb_arfcn','arfcn_dl','carrier','carrier_id','frekans_kanali'],
    'sector': ['sector','sector_id','sektor','sektör','sector_no','sektor_no','sektör_no','sektor_id','sektör_id'],
    'tac': ['tac','tracking_area_code','lac'],
    'prach_config_index': ['prach_config_index','prachconfigindex','prach_config','prachconfig',
                           'prach_configuration_index','rachconfigindex'],
    'zero_correlation_zone': ['zero_correlation_zone','zerocorrelationzoneconfig','zcz',
                              'zerocorrelationzone','ncs_config',
                              'zero_correlation_zone_config','prach_cs','prachcs','prach_ncs'],
    'high_speed': ['high_speed','highspeedflag','high_speed_flag','highspeed',
                   'restricted_set','restrictedset','restricted_set_config',
                   'prach_high_speed','speed_flag','hsflag'],
    'cell_range': ['cell_range','cellrange','cell_radius','cellradius',
                   'huawei_cell_range','huawei_cellrange','cell_range_m',
                   'cell_radius_m','max_cell_range','coverage_radius'],
}

# Column aliases for external neighbor list (attempts / handover counts)
NEIGHBOR_ATTEMPT_ALIASES = [
    'attempts', 'attempt', 'ho_attempts', 'ho_attempt',
    'handover', 'handovers', 'ho_count', 'ho',
    'att', 'deneme', 'girisim', 'sayi',
]

REQUIRED_COLUMNS = ['cell_id', 'latitude', 'longitude', 'pci']


def _detect_tech(df):
    """Guess the technology from the data.

    **Not used for any decision.**  The UI selection is the sole authority for
    technology (see read_excel_file); this helper only exists so a caller can
    surface an informational hint if it wants one.
    """
    if 'technology' in df.columns:
        vals = df['technology'].dropna().astype(str).str.upper().str.strip()
        if vals.str.contains('NR|5G').any():
            return 'NR'
    if 'pci' in df.columns and df['pci'].dropna().max() > 503:
        return 'NR'
    return 'LTE'


def normalize_column_names(df):
    df_norm = df.copy()
    col_lower = {c: c.strip().lower().replace(' ','_').replace('-','_') for c in df.columns}
    rename, matched = {}, []
    for std, aliases in COLUMN_ALIASES.items():
        for orig, low in col_lower.items():
            if low in aliases and orig not in rename:
                rename[orig] = std; matched.append(std); break
    unmatched = [c for c in df.columns if c not in rename]
    return df_norm.rename(columns=rename), list(set(matched)), unmatched


def validate_data(df, technology='LTE'):
    """Validate cell data against the technology selected in the UI.

    technology: 'LTE' (PCI 0-503, TS 36.211 §6.11)
                'NR'  (PCI 0-1007, TS 38.211 §7.4.2.1)
    The value comes from the UI, never from the file.
    """
    from pci_engine import norm_tech
    technology = norm_tech(technology)
    errors = []
    for c in REQUIRED_COLUMNS:
        if c not in df.columns: errors.append(f"Gerekli sütun eksik: '{c}'")
    if errors: return False, errors
    if df['cell_id'].isna().any():
        errors.append(f"{df['cell_id'].isna().sum()} satırda cell_id boş")
    d = df['cell_id'].duplicated().sum()
    if d > 0: errors.append(f"{d} tekrarlanan cell_id bulundu")
    if 'latitude' in df.columns:
        iv = ((df['latitude']<-90)|(df['latitude']>90)).sum()
        if iv > 0: errors.append(f"{iv} geçersiz latitude")
    if 'longitude' in df.columns:
        iv = ((df['longitude']<-180)|(df['longitude']>180)).sum()
        if iv > 0: errors.append(f"{iv} geçersiz longitude")
    if 'pci' in df.columns:
        max_pci = 1007 if technology == 'NR' else 503
        iv = ((df['pci'].dropna()<0)|(df['pci'].dropna()>max_pci)).sum()
        if iv > 0: errors.append(f"{iv} geçersiz PCI (0-{max_pci}, {technology})")
    if 'azimuth' in df.columns:
        iv = ((df['azimuth'].dropna()<0)|(df['azimuth'].dropna()>360)).sum()
        if iv > 0: errors.append(f"{iv} geçersiz azimuth (0-360)")
    if 'prach_config_index' in df.columns:
        from pci_engine import prach_config_max
        max_pcfg = prach_config_max(technology)
        iv = ((df['prach_config_index'].dropna()<0)|(df['prach_config_index'].dropna()>max_pcfg)).sum()
        if iv > 0: errors.append(f"{iv} geçersiz PRACH Config Index (0-{max_pcfg}, {technology})")
    return len(errors)==0, errors


def read_excel_file(uploaded_file, technology='LTE'):
    """Read and validate a cell-data Excel file.

    `technology` comes from the UI selection and is the **sole authority**:
    it decides the PCI range used for validation and is stamped onto the
    'technology' column, overriding whatever the file says.  The file is
    never inspected to decide LTE vs NR.
    """
    from pci_engine import norm_tech, pci_max
    technology = norm_tech(technology)
    msgs = []
    try:
        if isinstance(uploaded_file, bytes):
            uploaded_file = io.BytesIO(uploaded_file)
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        msgs.append(f"✅ {len(df)} satır, {len(df.columns)} sütun okundu")
        df, matched, unmatched = normalize_column_names(df)
        msgs.append(f"✅ Eşleşen sütunlar: {', '.join(matched)}")
        if unmatched:
            msgs.append(f"ℹ️ Eşleşmeyen sütunlar: {', '.join(unmatched)}")
        # --- Clean coordinate columns (handle comma decimals, DMS, etc.) ---
        for coord_col in ('latitude', 'longitude'):
            if coord_col in df.columns:
                original_dtype = df[coord_col].dtype
                df[coord_col] = _clean_coordinate_column(df[coord_col])
                bad_count = df[coord_col].isna().sum()
                if original_dtype == object:  # was string → likely needed conversion
                    msgs.append(f"ℹ️ {coord_col}: metin → ondalık derece dönüşümü uygulandı")
                if bad_count > 0:
                    msgs.append(f"⚠️ {coord_col}: {bad_count} satır ayrıştırılamadı")
        if 'azimuth' not in df.columns:
            df['azimuth'] = 0; msgs.append("⚠️ Azimuth yok, varsayılan 0°")
        if 'beamwidth' not in df.columns:
            df['beamwidth'] = 65.0
        # UI selection wins — stamp it on every row so exports and downstream
        # code can never disagree with the run's technology.  The file's own
        # label is only echoed back as a non-blocking note when it differs.
        _file_label = None
        if 'technology' in df.columns:
            _vals = df['technology'].dropna().astype(str).str.upper().str.strip()
            if len(_vals) > 0:
                _file_label = 'NR' if _vals.str.contains('NR|5G').any() else 'LTE'
        df['technology'] = technology
        if _file_label is not None and _file_label != technology:
            msgs.append(f"ℹ️ Dosyadaki teknoloji etiketi '{_file_label}' — arayüzde "
                        f"{technology} seçili olduğu için analiz {technology} "
                        f"kurallarıyla çalışacak (PCI 0-{pci_max(technology)}).")
        if 'prach_config_index' not in df.columns:
            df['prach_config_index'] = 0
            msgs.append("ℹ️ PRACH Config Index yok, varsayılan 0")
        if 'zero_correlation_zone' not in df.columns:
            df['zero_correlation_zone'] = 5
            if 'cell_range' not in df.columns:
                msgs.append("ℹ️ zeroCorrelationZoneConfig yok, varsayılan 5 (Ncs=26)")
        if 'cell_range' in df.columns:
            msgs.append("ℹ️ cellRange/cellRadius sütunu algılandı (Huawei modu)")
        if 'high_speed' in df.columns:
            from pci_engine import _row_restricted
            _hs_count = int(df.apply(_row_restricted, axis=1).sum())
            msgs.append(f"ℹ️ highSpeedFlag sütunu algılandı — {_hs_count} hücre "
                        f"restricted (high-speed) Ncs tablosu kullanacak")
        df['cell_id'] = df['cell_id'].astype(str)
        # Validate against the UI-selected technology, not a guess from the data.
        ok, errs = validate_data(df, technology=technology)
        if not ok:
            for e in errs: msgs.append(f"❌ {e}")
            return None, msgs
        msgs.append(f"✅ Veri doğrulaması başarılı ({technology}, PCI 0-{pci_max(technology)})")
        if 'pci' in df.columns:
            msgs.append(f"📊 Benzersiz PCI: {df['pci'].dropna().nunique()}")
        return df, msgs
    except Exception as e:
        msgs.append(f"❌ Dosya okuma hatası: {str(e)}")
        return None, msgs


def generate_sample_excel(technology='LTE'):
    """Generate a sample Excel with technology-specific example data.
    technology: 'LTE' or 'NR'
    """
    is_nr = (technology == 'NR')
    tech_label = 'NR' if is_nr else 'LTE'

    data = {
        'cell_id': [f'SITE{s:03d}_{c}' for s in range(1,7) for c in range(1,4)],
        'site_id': [f'SITE{s:03d}' for s in range(1,7) for _ in range(3)],
        'latitude': [
            41.0082, 41.0082, 41.0082,
            41.0102, 41.0102, 41.0102,
            41.0062, 41.0062, 41.0062,
            41.0122, 41.0122, 41.0122,
            41.0042, 41.0042, 41.0042,
            41.0092, 41.0092, 41.0092],
        'longitude': [
            28.9784, 28.9784, 28.9784,
            28.9804, 28.9804, 28.9804,
            28.9764, 28.9764, 28.9764,
            28.9824, 28.9824, 28.9824,
            28.9744, 28.9744, 28.9744,
            28.9794, 28.9794, 28.9794],
        'azimuth': [0,120,240, 0,120,240, 30,150,270, 60,180,300, 0,120,240, 10,130,250],
        'beamwidth': [65]*18,
        'sector': [1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3],
        'technology': [tech_label]*18,
        'tac': [1001]*9 + [1002]*9,
    }

    if is_nr:
        # NR: PCI 0-1007, band n78 (3500 MHz), PRACH config 0-255
        data['pci'] = [0,1,2, 3,0,5, 504,505,506, 750,1,800, 900,901,902, 15,16,2]
        data['rsi'] = [0,10,20, 30,0,50, 60,70,80, 90,100,110, 120,130,140, 150,160,170]
        data['band'] = [3500]*18
        data['prach_config_index'] = [0]*9 + [30]*9   # 0-27=long, ≥28=short
        data['zero_correlation_zone'] = [5,5,5, 5,5,5, 8,8,8, 3,3,3, 5,5,5, 8,8,8]
        data['cell_range'] = [None]*18  # Huawei: cellRadius (m), None=Nokia modu
    else:
        # LTE: PCI 0-503, band 1800 MHz, PRACH config 0-63
        data['pci'] = [0,1,2, 3,0,5, 6,7,8, 9,1,11, 12,13,14, 15,16,2]
        data['rsi'] = [0,10,20, 30,0,50, 60,70,80, 90,100,110, 120,130,140, 150,160,170]
        data['band'] = [1800]*18
        data['prach_config_index'] = [0]*18
        data['zero_correlation_zone'] = [5,5,5, 5,5,5, 8,8,8, 11,11,11, 5,5,5, 8,8,8]
        data['cell_range'] = [None]*18  # Huawei: cellRadius (m), None=Nokia modu

    df = pd.DataFrame(data)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as w:
        df.to_excel(w, sheet_name='CellData', index=False)

        # --- Format info sheet (technology-specific) ---
        if is_nr:
            pci_desc = 'Physical Cell ID (NR: 0-1007)'
            rsi_desc = 'Root Sequence Index (L=839: 0-837, L=139: 0-137)'
            prach_desc = 'PRACH Config Index (0-255; 0-27=long, ≥28=short sequence)'
            pci_ex, rsi_ex, prach_ex, band_ex = '750', '42', '0 veya 30', '3500'
        else:
            pci_desc = 'Physical Cell ID (LTE: 0-503)'
            rsi_desc = 'Root Sequence Index (0-837)'
            prach_desc = 'PRACH Config Index (0-63, preamble format belirler)'
            pci_ex, rsi_ex, prach_ex, band_ex = '150', '42', '0', '1800'

        info = {
            'Sütun': ['cell_id','site_id','latitude','longitude','azimuth','beamwidth',
                      'sector','pci','rsi','technology','band','tac','prach_config_index','zero_correlation_zone','cell_range'],
            'Zorunlu': ['Evet','Hayır','Evet','Evet','Hayır','Hayır','Hayır',
                       'Evet','Hayır','Hayır','Hayır','Hayır','Hayır','Hayır','Hayır'],
            'Açıklama': [
                'Benzersiz hücre ID','Site ID','Enlem (-90~90)','Boylam (-180~180)',
                'Anten yönü (0-360°)','Yatay hüzme genişliği (°)',
                'Sektör numarası (dolu ise naming convention yerine bu kullanılır)',
                pci_desc, rsi_desc,
                'LTE / NR','Frekans bandı (MHz)','Tracking Area Code',
                prach_desc,
                'zeroCorrelationZoneConfig (0-15, Ncs cyclic shift belirler → cell range)',
                'Huawei cellRadius (metre). Varsa ZCZ yerine bu değer kullanılır'],
            'Örnek': ['SITE001_1','SITE001','41.0082','28.9784','120','65',
                     '1', pci_ex, rsi_ex, tech_label, band_ex, '1001', prach_ex, '5', '14500']
        }
        pd.DataFrame(info).to_excel(w, sheet_name='Format Bilgisi', index=False)
    return out.getvalue()


# Maximum rows per sheet to prevent MemoryError with xlsxwriter
_EXCEL_MAX_ROWS = 50_000

def export_results_to_excel(df, results, neighbor_table, prach_info_df=None,
                            pci_suggestions=None, rsi_suggestions=None):
    from pci_engine import enrich_df_with_sector_info

    def _safe_write(frame, writer, sheet, max_rows=_EXCEL_MAX_ROWS):
        """Write DataFrame to excel, truncating to max_rows to avoid MemoryError."""
        if len(frame) > max_rows:
            frame = frame.head(max_rows).copy()
            # Add a note row indicating truncation
            note = {col: '' for col in frame.columns}
            first_col = frame.columns[0]
            note[first_col] = f'... toplam {len(frame)} satırdan ilk {max_rows} gösterildi'
            frame = pd.concat([frame, pd.DataFrame([note])], ignore_index=True)
        frame.to_excel(writer, sheet_name=sheet, index=False)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as w:
        enrich_df_with_sector_info(df).to_excel(w, sheet_name='Hücre Verileri', index=False)
        if prach_info_df is not None and len(prach_info_df) > 0:
            prach_info_df.to_excel(w, sheet_name='PRACH Bilgileri', index=False)
        if pci_suggestions is not None and len(pci_suggestions) > 0:
            pci_suggestions.to_excel(w, sheet_name='PCI Önerileri', index=False)
        if rsi_suggestions is not None and len(rsi_suggestions) > 0:
            rsi_suggestions.to_excel(w, sheet_name='RSI Önerileri', index=False)
        if len(neighbor_table) > 0:
            _safe_write(neighbor_table, w, 'Komşuluk Tablosu')
        for key, name in [('collisions','PCI Collision'),('confusions','PCI Confusion'),
                          ('mod3_conflicts','Mod3 Conflict'),('mod4_conflicts','Mod4 Conflict'),
                          ('mod6_conflicts','Mod6 Conflict'),
                          ('mod30_conflicts','Mod30 Conflict'),('rsi_collisions','RSI Collision')]:
            tbl = results.get(key, pd.DataFrame())
            if tbl is not None and len(tbl) > 0:
                _safe_write(tbl, w, name)
        sdf = pd.DataFrame([results['summary']]).T
        sdf.columns = ['Değer']; sdf.index.name = 'Metrik'
        sdf.to_excel(w, sheet_name='Özet')
    return out.getvalue()
