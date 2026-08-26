"""
Nokia OSS (RAML 2.0) PCI/RSI plan disa aktarimi
================================================
Uretilen dosya, operatorun mevcut OSS export'uyla ayni yapidadir:

    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE raml SYSTEM 'raml20.dtd'>
    <raml version="2.0" xmlns="raml20.xsd">
      <cmData type="plan" scope="all" name="...">
        <header><log dateTime="..." action="created" appInfo="..."/></header>
        <managedObject class="NRCELL" version="LN2.0"
                       distName="PLMN-PLMN/MRBTS-550016/NRBTS-1550016/NRCELL-211"
                       id="299350421" operation="update">
          <p name="physCellId">59</p>
          <p name="prachRootSequenceIndex">88</p>
        </managedObject>
        ...

HUCRE KIMLIGI — bu modulun tek kati kurali
------------------------------------------
OSS hucreyi `distName` ile tanir, planlama verisi `cell_id` ile.  Bu ikisi
arasinda TAHMINE dayali bir kopru kurulmaz: yanlis esleme, canli agda yanlis
hucreye PCI yazmak demektir ve dosyayi uretirken degil, sahada anlasilir.

Kabul edilen iki kaynak:

1. Hucre verisinde `dist_name` sutunu (ya da `mrbts` + `nrcell`, bkz.
   build_dist_name).  En saglam yol.
2. Sablon olarak yuklenen mevcut OSS XML'i — `id` ozniteligi buradan alinir ve
   `dist_name` dogrulanir.  Sablon TEK BASINA yeterli degildir: icinde hucre
   adi yoksa hangi satirin hangi hucre oldugunu soyleyemez.

Eslesmeyen hucre sessizce atlanmaz; rapor edilir ve dosyaya yazilmaz.
"""
import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd

# Dogrulanmis NR degerleri operatorun kendi export'undan alinmistir.
# LTE icin Nokia farkli sinif/parametre adlari kullanir; bir LTE ornegi
# gorulmeden varsayilan yapilmaz, cagiran tarafin vermesi beklenir.
NR_PROFILE = {
    'mo_class': 'NRCELL',
    'mo_version': 'LN2.0',
    'pci_param': 'physCellId',
    'rsi_param': 'prachRootSequenceIndex',
}

DIST_NAME_ALIASES = ['dist_name', 'distname', 'dn', 'distinguished_name',
                     'fdn', 'moid', 'mo_dn']


def find_dist_name_column(columns):
    norm = {str(c).strip().lower().replace(' ', '_').replace('-', '_'): c
            for c in columns}
    for a in DIST_NAME_ALIASES:
        if a in norm:
            return norm[a]
    return None


def build_dist_name(mrbts, nrcell, nrbts=None, plmn='PLMN-PLMN'):
    """distName from its parts.

    NRBTS defaults to 1000000 + MRBTS, which held for every object in the
    reference export — but it is only a default: pass nrbts explicitly when
    the network does not follow it.
    """
    mrbts = int(mrbts)
    if nrbts is None:
        nrbts = 1000000 + mrbts
    return f"{plmn}/MRBTS-{mrbts}/NRBTS-{int(nrbts)}/NRCELL-{int(nrcell)}"


def parse_raml(source):
    """Read a RAML 2.0 export.

    Returns (objects, meta):
      objects: distName -> {'id', 'class', 'version', 'operation', 'props'}
      meta:    {'count', 'classes', 'params', 'name'}
    """
    if isinstance(source, bytes):
        text = source.decode('utf-8', errors='replace')
    elif hasattr(source, 'read'):
        raw = source.read()
        text = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
    else:
        text = str(source)
    # DTD ve varsayilan ad alani ElementTree'yi zorlastirir; ikisini de kaldir
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, count=1)
    text = re.sub(r'\sxmlns="[^"]*"', '', text, count=1)
    root = ET.fromstring(text)

    objects = {}
    classes, params = {}, {}
    for mo in root.findall('.//managedObject'):
        dn = mo.get('distName')
        if not dn:
            continue
        props = {p.get('name'): (p.text or '').strip() for p in mo.findall('p')}
        objects[dn] = {'id': mo.get('id'), 'class': mo.get('class'),
                       'version': mo.get('version'),
                       'operation': mo.get('operation', 'update'),
                       'props': props}
        classes[mo.get('class')] = classes.get(mo.get('class'), 0) + 1
        for k in props:
            params[k] = params.get(k, 0) + 1
    cm = root.find('.//cmData')
    meta = {'count': len(objects), 'classes': classes, 'params': params,
            'name': cm.get('name') if cm is not None else None}
    return objects, meta


def resolve_cells(df, pci_col='pci', rsi_col='rsi', template_objects=None):
    """Match planning rows to OSS managed objects.

    Returns (resolved, unresolved):
      resolved:   list of dicts — cell_id, dist_name, id, pci, rsi
      unresolved: list of dicts — cell_id, reason
    """
    dn_col = find_dist_name_column(df.columns)
    resolved, unresolved = [], []

    have_parts = all(c in {str(x).strip().lower() for x in df.columns}
                     for c in ('mrbts', 'nrcell'))
    lower = {str(c).strip().lower(): c for c in df.columns}

    for _, r in df.iterrows():
        cid = str(r['cell_id'])
        dn = None
        if dn_col is not None:
            v = r.get(dn_col)
            if v is not None and not pd.isna(v) and str(v).strip():
                dn = str(v).strip()
        if dn is None and have_parts:
            m, c = r.get(lower['mrbts']), r.get(lower['nrcell'])
            if not (pd.isna(m) or pd.isna(c)):
                nb = r.get(lower.get('nrbts')) if 'nrbts' in lower else None
                dn = build_dist_name(m, c, None if nb is None or pd.isna(nb) else nb)
        if dn is None:
            unresolved.append({'cell_id': cid,
                               'reason': 'dist_name / mrbts+nrcell sutunu yok'})
            continue

        mo_id = None
        if template_objects is not None:
            if dn not in template_objects:
                unresolved.append({'cell_id': cid,
                                   'reason': f'sablonda bulunamadi: {dn}'})
                continue
            mo_id = template_objects[dn]['id']

        pci = r.get(pci_col)
        rsi = r.get(rsi_col)
        if pci is None or pd.isna(pci):
            unresolved.append({'cell_id': cid, 'reason': 'PCI degeri yok'})
            continue
        resolved.append({'cell_id': cid, 'dist_name': dn, 'id': mo_id,
                         'pci': int(pci),
                         'rsi': None if rsi is None or pd.isna(rsi) else int(rsi)})
    return resolved, unresolved


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def build_raml(resolved, file_name='PCI_RSI_Plan.xml', profile=None,
               app_info='TurkTelekom PCI/RSI Planner v3',
               include_rsi=True, operation='update'):
    """Write the RAML 2.0 plan file. Returns bytes."""
    prof = dict(NR_PROFILE)
    if profile:
        prof.update(profile)
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    out = io.StringIO()
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.write("<!DOCTYPE raml SYSTEM 'raml20.dtd'>\n")
    out.write('<raml version="2.0" xmlns="raml20.xsd">\n')
    out.write(f'  <cmData type="plan" scope="all" name="{_esc(file_name)}">\n')
    out.write('    <header>\n')
    out.write(f'      <log dateTime="{now}" action="created" '
              f'appInfo="{_esc(app_info)}"/>\n')
    out.write('    </header>\n')
    for c in resolved:
        id_attr = f' id="{_esc(c["id"])}"' if c.get('id') else ''
        out.write(f'    <managedObject class="{_esc(prof["mo_class"])}" '
                  f'version="{_esc(prof["mo_version"])}" '
                  f'distName="{_esc(c["dist_name"])}"{id_attr} '
                  f'operation="{_esc(operation)}">\n')
        out.write(f'      <p name="{_esc(prof["pci_param"])}">{int(c["pci"])}</p>\n')
        if include_rsi and c.get('rsi') is not None:
            out.write(f'      <p name="{_esc(prof["rsi_param"])}">'
                      f'{int(c["rsi"])}</p>\n')
        out.write('    </managedObject>\n')
    out.write('  </cmData>\n')
    out.write('</raml>\n')
    return out.getvalue().encode('utf-8')


def diff_against_template(resolved, template_objects, profile=None):
    """What would actually change in the OSS. Returns a DataFrame.

    An export that rewrites values the network already has is noise; this is
    what makes the plan reviewable before it is applied.
    """
    prof = dict(NR_PROFILE)
    if profile:
        prof.update(profile)
    rows = []
    for c in resolved:
        cur = (template_objects or {}).get(c['dist_name'], {}).get('props', {})
        cur_pci = cur.get(prof['pci_param'])
        cur_rsi = cur.get(prof['rsi_param'])
        cur_pci_i = int(cur_pci) if cur_pci not in (None, '') else None
        cur_rsi_i = int(cur_rsi) if cur_rsi not in (None, '') else None
        rows.append({
            'cell_id': c['cell_id'], 'dist_name': c['dist_name'],
            'PCI (OSS)': cur_pci_i, 'PCI (plan)': c['pci'],
            'PCI değişti': '✅' if cur_pci_i != c['pci'] else '',
            'RSI (OSS)': cur_rsi_i, 'RSI (plan)': c['rsi'],
            'RSI değişti': '✅' if (c['rsi'] is not None
                                    and cur_rsi_i != c['rsi']) else '',
        })
    return pd.DataFrame(rows)
