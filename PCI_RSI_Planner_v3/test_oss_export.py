"""
Nokia OSS (RAML 2.0) disa aktarimi — uctan uca dogrulama
=========================================================
Sablon olarak operatorun GERCEK OSS export'lari kullanilir:
  NR  : RanG_Samsun_PCI_RSI_Int_update.xml            (925 NRCELL)
  LTE : ..._1cell_LTE_phyCellId_rootSeqIndex_...xml   (LNCEL + LNCEL_FDD)

Calistirma:  python test_oss_export.py
"""
import glob
import os
import re
import sys

import pandas as pd

from nokia_export import (parse_raml, build_raml, resolve_cells,
                          diff_against_template, profile_for, child_dist_name)

BASE = r"C:\Users\PCUser\Desktop\PCI_RSI_Planları"
_fails = []


def check(cond, msg):
    if cond:
        print(f"  OK   {msg}")
    else:
        print(f"  FAIL {msg}")
        _fails.append(msg)


def find(pattern):
    hits = glob.glob(os.path.join(BASE, pattern))
    return hits[0] if hits else None


# ============================================================
print("\n=== 1. Profiller ===")
pn, pl = profile_for('NR'), profile_for('LTE')
check(pn['mo_class'] == 'NRCELL' and pn['pci_param'] == 'physCellId'
      and pn['rsi_param'] == 'prachRootSequenceIndex' and not pn['rsi_child_class'],
      "NR: NRCELL, physCellId + prachRootSequenceIndex, tek nesne")
check(pl['mo_class'] == 'LNCEL' and pl['pci_param'] == 'phyCellId'
      and pl['rsi_param'] == 'rootSeqIndex' and pl['rsi_child_class'] == 'LNCEL_FDD',
      "LTE: LNCEL.phyCellId + LNCEL_FDD.rootSeqIndex")
check(profile_for('LTE', 'TDD')['rsi_child_class'] == 'LNCEL_TDD',
      "LTE TDD: cocuk sinif LNCEL_TDD")
check(pn['bts_offset'] == 1000000 and pl['bts_offset'] == 0,
      "NRBTS = 1000000+MRBTS, LNBTS = MRBTS")

# ============================================================
nr_file = find("RanG_Samsun*.xml")
if nr_file:
    print("\n=== 2. NR: gercek export ile round-trip ===")
    raw = open(nr_file, 'rb').read()
    objs, meta = parse_raml(raw)
    check(meta['guessed_tech'] == 'NR', f"teknoloji NR olarak taninadi: {meta['classes']}")
    res = [{'cell_id': dn, 'dist_name': dn, 'id': o['id'],
            'child_dist_name': None, 'child_id': None,
            'pci': int(o['props']['physCellId']),
            'rsi': int(o['props']['prachRootSequenceIndex']),
            'profile': pn} for dn, o in objs.items()]
    # cmData/@name sablondaki degerdir; dosya adi ondan farkli olabilir
    out = build_raml(res, file_name=meta['name'], app_info='PlanExporter')
    a = raw.decode('utf-8').splitlines()
    b = out.decode('utf-8').splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    check(len(a) == len(b), f"satir sayisi ayni ({len(a)} / {len(b)})")
    check(len(diff) <= 1, f"yalnizca zaman damgasi farkli ({len(diff)} satir)")
    back, _ = parse_raml(out)
    check(all(back[k]['props'] == objs[k]['props'] and back[k]['id'] == objs[k]['id']
              for k in objs), "tum PCI/RSI/id degerleri korunuyor")

lte_file = find("*LTE_phyCellId*.xml")
if lte_file:
    print("\n=== 3. LTE: gercek export ile round-trip ===")
    raw = open(lte_file, 'rb').read()
    objs, meta = parse_raml(raw)
    check(meta['guessed_tech'] == 'LTE', f"teknoloji LTE olarak taninadi: {meta['classes']}")
    parents = {dn: o for dn, o in objs.items() if o['class'] == 'LNCEL'}
    check(len(parents) >= 1, f"{len(parents)} LNCEL nesnesi")
    res = []
    for dn, o in parents.items():
        cdn = child_dist_name(dn, pl)
        res.append({'cell_id': dn, 'dist_name': dn, 'id': o['id'],
                    'child_dist_name': cdn,
                    'child_id': objs.get(cdn, {}).get('id'),
                    'pci': int(o['props']['phyCellId']),
                    'rsi': int(objs[cdn]['props']['rootSeqIndex']),
                    'profile': pl})
    out = build_raml(res, file_name=meta['name'], app_info='PlanExporter')
    a = raw.decode('utf-8').splitlines()
    b = out.decode('utf-8').splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    check(len(a) == len(b), f"satir sayisi ayni ({len(a)} / {len(b)})")
    check(len(diff) <= 1, f"yalnizca zaman damgasi farkli ({len(diff)} satir)")
    back, _ = parse_raml(out)
    check(set(back) == set(objs), "ayni distName kumesi")
    check(all(back[k]['props'] == objs[k]['props'] and back[k]['id'] == objs[k]['id']
              for k in objs), "tum PCI/RSI/id degerleri korunuyor")

    print("\n=== 4. LTE: veriden uretim (dist_name sutunu) ===")
    dn0 = list(parents)[0]
    d = pd.DataFrame({'cell_id': ['TEST1'], 'dist_name': [dn0],
                      'pci': [123], 'rsi': [456]})
    r, u = resolve_cells(d, technology='LTE', template_objects=objs)
    check(len(r) == 1 and not u, "hucre eslesti")
    check(r[0]['child_dist_name'] == child_dist_name(dn0, pl), "cocuk distName kuruldu")
    check(r[0]['child_id'] == objs[r[0]['child_dist_name']]['id'],
          "cocuk id sablondan alindi")
    x = build_raml(r).decode('utf-8')
    check(x.count('<managedObject') == 2, "hucre basina IKI managedObject")
    check('phyCellId">123<' in x and 'rootSeqIndex">456<' in x,
          "PCI parent'ta, RSI cocukta")
    check('LNCEL_FDD' in x, "cocuk sinifi LNCEL_FDD")

    print("\n=== 5. LTE: RSI kapatilinca cocuk nesne yazilmamali ===")
    x2 = build_raml(r, include_rsi=False).decode('utf-8')
    check(x2.count('<managedObject') == 1, "yalnizca LNCEL yazildi")
    check('rootSeqIndex' not in x2, "rootSeqIndex yok")

# ============================================================
print("\n=== 6. Kimlik yoksa hicbir sey uretilmemeli ===")
d3 = pd.DataFrame({'cell_id': ['A', 'B'], 'pci': [1, 2], 'rsi': [3, 4]})
r3, u3 = resolve_cells(d3, technology='NR')
check(not r3 and len(u3) == 2, f"0 uretildi, {len(u3)} sebep raporlandi")
check(u3[0]['reason'] == 'dist_name yok', f"sebep: {u3[0]['reason']}")

if nr_file:
    print("\n=== 7. Sablonda olmayan / eksik veri atlanmali ===")
    objs, _ = parse_raml(open(nr_file, 'rb').read())
    d4 = pd.DataFrame({'cell_id': ['X'],
                       'dist_name': ['PLMN-PLMN/MRBTS-9/NRBTS-9/NRCELL-9'],
                       'pci': [5], 'rsi': [6]})
    r4, u4 = resolve_cells(d4, technology='NR', template_objects=objs)
    check(not r4 and 'şablonda bulunamadı' in u4[0]['reason'],
          f"bilinmeyen distName atlandi: {u4[0]['reason']}")
    dn0 = list(objs)[0]
    d5 = pd.DataFrame({'cell_id': ['Y'], 'dist_name': [dn0], 'pci': [None], 'rsi': [1]})
    r5, u5 = resolve_cells(d5, technology='NR', template_objects=objs)
    check(not r5 and 'PCI' in u5[0]['reason'], f"PCI'si bos satir atlandi")

    print("\n=== 8. distName parcalardan KURULMAZ ===")
    # mrbts + hucre no verilse bile distName uydurulmaz.  Bir esleme kurali
    # tahmin etmek, kuralin gecmedigi tek hucrede plani yanlis nesneye yazar
    # ve bu dosya uretilirken degil sahada anlasilir.
    dn_any = list(objs)[0]
    d6 = pd.DataFrame({
        'cell_id': ['NOPARTS'],
        'mrbts': [int(re.search(r'MRBTS-(\d+)', dn_any).group(1))],
        'nrcell': [int(re.search(r'NRCELL-(\d+)', dn_any).group(1))],
        'pci': [1], 'rsi': [2]})
    r6, u6 = resolve_cells(d6, technology='NR', template_objects=objs)
    check(not r6 and u6[0]['reason'] == 'dist_name yok',
          f"mrbts+nrcell varken bile uretilmedi: {u6[0]['reason']}")

    print("\n=== 9. Sablonsuz uretimde id yazilmaz ===")
    r7, _ = resolve_cells(pd.DataFrame({'cell_id': ['Z'], 'dist_name': [dn0],
                                        'pci': [1], 'rsi': [2]}), technology='NR')
    check(' id=' not in build_raml(r7).decode('utf-8'), "id ozniteligi yok")

# ============================================================
print("\n" + "=" * 60)
if _fails:
    print(f"{len(_fails)} TEST BASARISIZ:")
    for f in _fails:
        print("  -", f)
    sys.exit(1)
print("Tum OSS testleri gecti.")
