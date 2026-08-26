# PCI / RSI Planner v2 — dondurulmuş

Bu sürüm **değiştirilmez.** Karşılaştırma ve geri dönüş için duruyor.

```
run.bat  ->  http://localhost:8501
```

Sanal ortam üst klasörde paylaşılır; önce `..\setup.bat` çalıştırın.

## Neden dondurulmuş

2026-08-26'da yapılan 3GPP uygunluk denetiminde bu sürümde 17 bulgu çıktı;
hiçbiri burada düzeltilmedi. Gerçek bir NR ağında ölçülen başlıcaları:

| Bulgu | v2'deki durum |
|---|---|
| **Teknoloji otoritesi** | Kenar çubuğu NR seçiliyken bile NR verisi LTE olarak koşabiliyor; plana 503 üstü PCI sızabiliyor |
| **Taşıyıcı kapsamı** | Farklı ARFCN'deki hücreler çakışma sayılıyor — 926 hücrelik ağda bulguların %62'si sahte |
| **Kısıtlı küme Ncs** | TS 36.211 Tablo 5.7.2-2'nin kısıtlı sütunu zcz=7'den itibaren yanlış (59/76/93… yerine 55/68/82…) |
| **Kısıtlı küme kök sayısı** | `floor(Nzc/Ncs)` kullanılıyor; Ncs=76'da 6 kök ayrılıyor, standart 32 istiyor |
| **LTE format 4** | `get_ncs` LTE'de `short` parametresini yok sayıyor |
| **PRACH config → format** | FDD'de 58-63 format 4 sayılıyor; FDD'de format 4 yok |
| **Hücre menzili** | Toplam T_SEQ kullanılıyor; LTE format 2/3 iki kat, NR format 3 dört kat yüksek |
| **Komşuluk ızgarası** | Boylam kovası enleme göre ölçeklenmiyor; çiftlerin %3,4'ü sessizce kaçıyor |
| **Planlayıcı** | Girdiden kötü bir plan döndürebilir ve bunu fark etmez |

Hepsi ve düzeltilmiş hâlleri: [`../PCI_RSI_Planner_v3/`](../PCI_RSI_Planner_v3/)

## Test

```
python test_engine.py
```

Not: bu takım `get_lte_preamble_format(60) == 4` iddiasını içerir — yani yukarıdaki
PRACH config hatasını kendi testiyle sabitlemiştir. v3'te hem kod hem test
düzeltildi.
