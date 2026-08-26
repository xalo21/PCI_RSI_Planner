"""
Nokia OSS (RAML 2.0) PCI/RSI plan disa aktarimi
================================================
Uretilen dosya, operatorun kendi OSS export'uyla ayni yapidadir.  LTE ve NR'nin
yapisi FARKLI oldugu icin iki profil vardir — ikisi de operatorun gercek
export dosyalarindan dogrulanmistir.

NR — hucre basina TEK nesne, PCI ve RSI ayni nesnede:

    <managedObject class="NRCELL" version="LN2.0"
                   distName="PLMN-PLMN/MRBTS-550016/NRBTS-1550016/NRCELL-211"
                   id="299350421" operation="update">
      <p name="physCellId">59</p>
      <p name="prachRootSequenceIndex">88</p>
    </managedObject>

LTE — hucre basina IKI nesne, RSI cocuk nesnede ve AYRI bir id ile:

    <managedObject class="LNCEL" version="LN2.0"
                   distName="PLMN-PLMN/MRBTS-40309/LNBTS-40309/LNCEL-11"
                   id="1189780" operation="update">
      <p name="phyCellId">489</p>
    </managedObject>
    <managedObject class="LNCEL_FDD" version="LN2.0"
                   distName="PLMN-PLMN/MRBTS-40309/LNBTS-40309/LNCEL-11/LNCEL_FDD-0"
                   id="2328865" operation="update">
      <p name="rootSeqIndex">390</p>
    </managedObject>

Iki fark daha: LTE'de LNBTS = MRBTS, NR'de NRBTS = 1000000 + MRBTS; ve LTE'de
cocuk nesnenin sinifi duplex'e gore LNCEL_FDD / LNCEL_TDD olur.

HUCRE KIMLIGI — bu modulun tek kati kurali
------------------------------------------
OSS hucreyi `distName` ile tanir, planlama verisi `cell_id` ile.  Bu koprü
VERIDEN gelir: hucre dosyasinda `dist_name` sutunu olmak zorundadir.  Parcalardan
(MRBTS + hucre no) distName KURULMAZ ve hicbir eslestirme kurali tahmin
edilmez — boyle bir kuralin gecmedigi tek hucre, plani yanlis nesneye yazar ve
bu dosya uretilirken degil sahada anlasilir.

`dist_name` yoksa XML uretilmez.  Sutun varsa ama bir satirda bossa, o hucre
rapor edilir ve dosyaya yazilmaz.
"""
import io
import re
from datetime import datetime

import pandas as pd

# Her iki profil de operatorun gercek export dosyalarindan dogrulandi.
NR_PROFILE = {
    'tech': 'NR',
    'mo_class': 'NRCELL',
    'mo_version': 'LN2.0',
    'pci_param': 'physCellId',
    'rsi_param': 'prachRootSequenceIndex',
    'rsi_child_class': None,        # RSI ayni nesnede
    'bts_tag': 'NRBTS',
    'cell_tag': 'NRCELL',
    'bts_offset': 1000000,          # NRBTS = 1000000 + MRBTS
}

LTE_PROFILE = {
    'tech': 'LTE',
    'mo_class': 'LNCEL',
    'mo_version': 'LN2.0',
    'pci_param': 'phyCellId',
    'rsi_param': 'rootSeqIndex',
    'rsi_child_class': 'LNCEL_FDD',  # duplex'e gore LNCEL_TDD olabilir
    'rsi_child_index': 0,
    'bts_tag': 'LNBTS',
    'cell_tag': 'LNCEL',
    'bts_offset': 0,                 # LNBTS = MRBTS
}


def profile_for(technology, duplex=None):
    """Profile for the technology, with the LTE child class set by duplex."""
    t = str(technology).strip().upper()
    if 'NR' in t or '5G' in t:
        return dict(NR_PROFILE)
    p = dict(LTE_PROFILE)
    if duplex and str(duplex).strip().upper().startswith('T'):
        p['rsi_child_class'] = 'LNCEL_TDD'
    return p


DIST_NAME_ALIASES = ['dist_name', 'distname', 'dn', 'distinguished_name',
                     'fdn', 'moid', 'mo_dn']


def find_dist_name_column(columns):
    norm = {str(c).strip().lower().replace(' ', '_').replace('-', '_'): c
            for c in columns}
    for a in DIST_NAME_ALIASES:
        if a in norm:
            return norm[a]
    return None


def child_dist_name(parent_dn, profile):
    """distName of the object carrying RSI, or None when it is the parent."""
    cls = profile.get('rsi_child_class')
    if not cls:
        return None
    return f"{parent_dn}/{cls}-{int(profile.get('rsi_child_index', 0))}"


def parse_raml(source):
    """Read a RAML 2.0 export.

    Returns (objects, meta):
      objects: distName -> {'id', 'class', 'version', 'operation', 'props'}
      meta:    {'count', 'classes', 'params', 'name', 'guessed_tech'}
    """
    if isinstance(source, bytes):
        text = source.decode('utf-8', errors='replace')
    elif hasattr(source, 'read'):
        raw = source.read()
        text = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
    else:
        text = str(source)
    # DTD ve varsayilan ad alani ElementTree'yi zorlastirir; ikisini de kaldir
    import xml.etree.ElementTree as ET
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
    guessed = ('NR' if any(c.startswith('NRCELL') for c in classes)
               else 'LTE' if any(c.startswith('LNCEL') for c in classes) else None)
    meta = {'count': len(objects), 'classes': classes, 'params': params,
            'name': cm.get('name') if cm is not None else None,
            'guessed_tech': guessed}
    return objects, meta


def resolve_cells(df, technology='NR', pci_col='pci', rsi_col='rsi',
                  template_objects=None):
    """Match planning rows to OSS managed objects.

    Returns (resolved, unresolved).  A resolved entry carries the parent
    distName/id and, for LTE, the child distName/id that holds rootSeqIndex.
    When a template is supplied the child is looked up in it rather than
    assumed, so an installation that does not use index 0 still works.
    """
    dn_col = find_dist_name_column(df.columns)
    resolved, unresolved = [], []

    for _, r in df.iterrows():
        cid = str(r['cell_id'])
        prof = profile_for(technology,
                           r.get('duplex') if 'duplex' in df.columns else None)

        # distName VERIDEN gelir; parcalardan kurulmaz.  Bir kural uydurup
        # distName insa etmek, o kuralin gecmedigi tek hucrede plani yanlis
        # nesneye yazar ve bu dosya uretilirken degil sahada anlasilir.
        dn = None
        if dn_col is not None:
            v = r.get(dn_col)
            if v is not None and not pd.isna(v) and str(v).strip():
                dn = str(v).strip()
        if dn is None:
            unresolved.append({'cell_id': cid, 'reason': 'dist_name yok'})
            continue

        parent_id = child_id = None
        child_dn = child_dist_name(dn, prof)
        if template_objects is not None:
            if dn not in template_objects:
                unresolved.append({'cell_id': cid,
                                   'reason': f'şablonda bulunamadı: {dn}'})
                continue
            parent_id = template_objects[dn]['id']
            if child_dn is not None:
                if child_dn in template_objects:
                    child_id = template_objects[child_dn]['id']
                else:
                    # index 0 varsayimi tutmadi — sablondaki gercek cocugu ara
                    pref = f"{dn}/{prof['rsi_child_class']}-"
                    cands = [k for k in template_objects if k.startswith(pref)]
                    if len(cands) == 1:
                        child_dn = cands[0]
                        child_id = template_objects[child_dn]['id']
                    else:
                        child_dn = None   # RSI yazilamaz, PCI yine de yazilir

        pci = r.get(pci_col)
        rsi = r.get(rsi_col) if rsi_col in df.columns else None
        if pci is None or pd.isna(pci):
            unresolved.append({'cell_id': cid, 'reason': 'PCI değeri yok'})
            continue
        resolved.append({
            'cell_id': cid, 'dist_name': dn, 'id': parent_id,
            'child_dist_name': child_dn, 'child_id': child_id,
            'pci': int(pci),
            'rsi': None if rsi is None or pd.isna(rsi) else int(rsi),
            'profile': prof})
    return resolved, unresolved


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def _mo(out, cls, version, dn, mo_id, operation, params):
    id_attr = f' id="{_esc(mo_id)}"' if mo_id else ''
    out.write(f'    <managedObject class="{_esc(cls)}" version="{_esc(version)}" '
              f'distName="{_esc(dn)}"{id_attr} operation="{_esc(operation)}">\n')
    for name, val in params:
        out.write(f'      <p name="{_esc(name)}">{val}</p>\n')
    out.write('    </managedObject>\n')


def build_raml(resolved, file_name='PCI_RSI_Plan.xml',
               app_info='TurkTelekom PCI/RSI Planner v3',
               include_rsi=True, operation='update'):
    """Write the RAML 2.0 plan file. Returns bytes."""
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
        prof = c.get('profile') or NR_PROFILE
        parent_params = [(prof['pci_param'], int(c['pci']))]
        write_rsi = include_rsi and c.get('rsi') is not None
        if write_rsi and not prof.get('rsi_child_class'):
            parent_params.append((prof['rsi_param'], int(c['rsi'])))
        _mo(out, prof['mo_class'], prof['mo_version'], c['dist_name'],
            c.get('id'), operation, parent_params)
        if write_rsi and prof.get('rsi_child_class') and c.get('child_dist_name'):
            _mo(out, prof['rsi_child_class'], prof['mo_version'],
                c['child_dist_name'], c.get('child_id'), operation,
                [(prof['rsi_param'], int(c['rsi']))])
    out.write('  </cmData>\n')
    out.write('</raml>\n')
    return out.getvalue().encode('utf-8')


def diff_against_template(resolved, template_objects):
    """What would actually change in the OSS. Returns a DataFrame.

    An export that rewrites values the network already has is noise; this is
    what makes the plan reviewable before it is applied.
    """
    tpl = template_objects or {}
    rows = []
    for c in resolved:
        prof = c.get('profile') or NR_PROFILE
        cur_pci = tpl.get(c['dist_name'], {}).get('props', {}).get(prof['pci_param'])
        rsi_dn = c.get('child_dist_name') or c['dist_name']
        cur_rsi = tpl.get(rsi_dn, {}).get('props', {}).get(prof['rsi_param'])
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
