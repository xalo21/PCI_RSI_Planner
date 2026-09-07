# TürkTelekom PCI / RSI Planner

3GPP TS 36.211 / 38.211 / 38.331'e göre 4G (LTE) ve 5G (NR) için PCI ve PRACH
Root Sequence Index planlama aracı.

İki sürüm ayrı klasörlerde, ayrı uygulamalar olarak durur ve **aynı anda
çalışabilir**:

```
PCI_RSI_Planner_v2/   run.bat      ->  http://localhost:8501   (dondurulmuş)
PCI_RSI_Planner_v3/   run_v3.bat   ->  http://localhost:8502   (güncel)
```

## Kurulum

### v3 — kendi kendine yeter

`PCI_RSI_Planner_v3/` klasörü **tek başına taşınabilir**: kurulumu, çevrimdışı
paketleri (`wheels/`), yapılandırması ve uygulaması hep içinde. Klasörü
kopyalayın, içinde:

```
setup.bat            ->  .venv'i aynı klasörde, çevrimdışı kurar
run_v3.bat           ->  internet varsa (harita çalışır)
run_v3_offline.bat   ->  kapalı ağ / kurum içi (harita kapalı)
```

İnternet erişimi olmayan bir kuruma dağıtım için:
[`PCI_RSI_Planner_v3/KURULUM_KAPALI_AG.md`](PCI_RSI_Planner_v3/KURULUM_KAPALI_AG.md)

### v2 — kökteki setup.bat

```
setup.bat
```

Kök klasörde kendi `.venv`'ini kurar. Çevrimdışı paketleri
`PCI_RSI_Planner_v3/wheels/` içinden okur — 161 MB'lık klasörün iki nüshasını
tutmuyoruz.

## Hangi sürümü kullanmalı

**v3.** v2 yalnızca karşılaştırma ve geri dönüş için duruyor; 2026-08-26'daki
3GPP uygunluk denetiminde bulunan hataların hiçbiri onda düzeltilmedi.

v3'te düzeltilenlerin başlıcaları:

- **Teknoloji ve PCI aralığı** — v2, NR verisini sessizce LTE olarak koşabiliyor
  ve plana geçersiz PCI yazabiliyordu
- **Taşıyıcı kapsamı** — çakışmalar yalnızca aynı ARFCN'deki hücreler arasında
  sayılır. Gerçek bir ağda bulguların %62'si bu yüzden sahte çıkmıştı
- **PRACH tabloları** — spec metninden çıkarıldı; v2'nin kısıtlı küme Ncs
  tablosu zcz=7'den itibaren yanlıştı
- **Hücre menzili** — Ncs penceresi `1/Δf_RA`; LTE format 2/3 iki kat, NR
  format 3 dört kat yanlış hesaplanıyordu
- **Komşuluk grafiği** — mekânsal ızgara enleme göre ölçekli; v2 doğu-batı
  yönünde çiftlerin %3,4'ünü sessizce kaçırıyordu

Ayrıntı ve ölçümler: [`PCI_RSI_Planner_v3/README.md`](PCI_RSI_Planner_v3/README.md)

## İki sürümü karşılaştırma

```
cd PCI_RSI_Planner_v3
python compare_v2_v3.py --data <hucreler.xlsx> --tech LTE
```

İki motoru ayrı süreçlerde aynı veriyle koşturup özet, komşuluk grafiği, hücre
bazlı PRACH parametreleri ve plan çıktılarını diff'ler. Bir planı devreye almadan
önce farkların hepsinin kasıtlı düzeltmelere karşılık geldiğini doğrulamak için.

## Depo düzeni

| Dal | İçerik |
|---|---|
| `master` | v2'nin dondurulmuş anlık görüntüsü, `v2.0` etiketiyle. Kök dizinde, tarihsel hâliyle |
| `v3-rewrite` | Güncel çalışma: her iki uygulama da kendi klasöründe |

`PCI_RSI_Planner_v3/wheels/` (~161 MB çevrimdışı paket önbelleği, Python 3.12 ve
3.14 için ayrı setler) ve `.venv/` sürüm kontrolüne dahil **değildir**.

> ⚠️ Bu yüzden hedef makineye `git clone` ile kurulum yapılamaz — klonda
> `wheels/` gelmez ve internetsiz ortamda paketler indirilemez. Dağıtım için
> klasörü dosya kopyası (USB vb.) ile taşıyın.
