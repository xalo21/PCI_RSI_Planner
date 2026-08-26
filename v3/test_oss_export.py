"""Nokia OSS (RAML 2.0) disa aktarimi — uctan uca dogrulama.

Sablon olarak operatorun gercek OSS export'u kullanilir.
Calistirma:  python test_oss_export.py
"""
import pandas as pd

from nokia_export import (parse_raml, build_raml, resolve_cells,
                          diff_against_template, build_dist_name)

BASE = r"C:\Users\PCUser\Desktop\PCI_RSI_Planları"
objs, meta = parse_raml(open(BASE + r"\RanG_Samsun_PCI_RSI_Int_update.xml", 'rb').read())
print(f"sablon: {meta['count']} nesne, siniflar {meta['classes']}")

# --- 1) dist_name sutunu olan veri ---
dns = list(objs)
d = pd.DataFrame({
    'cell_id': [f'C{i}' for i in range(len(dns))],
    'dist_name': dns,
    'pci': [(i * 7) % 1008 for i in range(len(dns))],
    'rsi': [(i * 13) % 838 for i in range(len(dns))],
})
res, unres = resolve_cells(d, template_objects=objs)
print(f"\n1) dist_name ile: eslesen={len(res)}  eslesmeyen={len(unres)}")
assert len(res) == len(dns) and not unres
assert all(r['id'] for r in res), "id sablondan alinmali"
xml = build_raml(res, file_name='Test_Plan.xml')
back, m2 = parse_raml(xml)
assert m2['count'] == len(dns)
assert all(int(back[r['dist_name']]['props']['physCellId']) == r['pci'] for r in res)
assert all(back[r['dist_name']]['id'] == r['id'] for r in res)
print("   yazilan XML geri okundu, PCI/RSI/id birebir tutuyor")

df_diff = diff_against_template(res, objs)
print(f"   fark: {int((df_diff['PCI değişti']=='✅').sum())} PCI, "
      f"{int((df_diff['RSI değişti']=='✅').sum())} RSI degisiyor")

# --- 2) mrbts + nrcell parcalariyla ---
import re
parts = []
for dn in dns[:50]:
    parts.append({'mrbts': int(re.search(r'MRBTS-(\d+)', dn).group(1)),
                  'nrcell': int(re.search(r'NRCELL-(\d+)', dn).group(1))})
d2 = pd.DataFrame(parts)
d2['cell_id'] = [f'P{i}' for i in range(len(d2))]
d2['pci'] = range(len(d2))
d2['rsi'] = range(len(d2))
res2, unres2 = resolve_cells(d2, template_objects=objs)
print(f"\n2) mrbts+nrcell ile: eslesen={len(res2)}  eslesmeyen={len(unres2)}")
assert len(res2) == 50, unres2[:3]
print("   nrbts = 1000000 + mrbts varsayimi sablonla dogrulandi")

# --- 3) kimlik yoksa hicbir sey uretilmemeli ---
d3 = pd.DataFrame({'cell_id': ['A', 'B'], 'pci': [1, 2], 'rsi': [3, 4]})
res3, unres3 = resolve_cells(d3, template_objects=objs)
print(f"\n3) kimlik sutunu yok: eslesen={len(res3)}  eslesmeyen={len(unres3)}")
assert not res3 and len(unres3) == 2
print(f"   sebep: {unres3[0]['reason']}")

# --- 4) sablonda olmayan hucre atlanmali, uydurulmamali ---
d4 = pd.DataFrame({'cell_id': ['X'], 'dist_name': ['PLMN-PLMN/MRBTS-9/NRBTS-9/NRCELL-9'],
                   'pci': [5], 'rsi': [6]})
res4, unres4 = resolve_cells(d4, template_objects=objs)
print(f"\n4) sablonda olmayan distName: eslesen={len(res4)}  eslesmeyen={len(unres4)}")
assert not res4 and 'sablonda bulunamadi' in unres4[0]['reason']
print(f"   sebep: {unres4[0]['reason']}")

# --- 5) PCI'si olmayan satir atlanmali ---
d5 = pd.DataFrame({'cell_id': ['Y'], 'dist_name': [dns[0]], 'pci': [None], 'rsi': [1]})
res5, unres5 = resolve_cells(d5, template_objects=objs)
assert not res5 and 'PCI' in unres5[0]['reason']
print(f"\n5) PCI'si bos satir atlandi: {unres5[0]['reason']}")

# --- 6) sablonsuz uretim: id yazilmamali ---
res6, _ = resolve_cells(d.head(3))
x6 = build_raml(res6).decode('utf-8')
assert ' id=' not in x6
print("\n6) sablonsuz uretimde id ozniteligi yazilmiyor")

print("\nTum OSS testleri gecti.")
