# Kapalı Ağda Kurulum (Air-Gapped / Kurum İçi)

İnternet erişimi olmayan bir makineye kurulum ve kurum içi erişim açma adımları.

---

## 1. Hedef makinede Python

**Python 3.12 x64** kurulu olmalı (3.14 x64 de olur).

> **Neden sürüm önemli:** `wheels/` klasöründeki derlenmiş paketler
> (`numpy`, `pandas`, `pyarrow`, `pillow`, `markupsafe`, `charset_normalizer`,
> `rpds_py`) Python sürümüne kilitlidir — dosya adındaki `cp312` / `cp314`
> etiketi bunu gösterir. Başka bir sürümde kurulum **kaçınılmaz olarak**
> başarısız olur, çünkü internet olmadığı için pip alternatif indiremez.
> Klasörde 3.12 ve 3.14 için ayrı setler var; `setup.bat` ikisini de kabul eder.

Python kurulumunda **"Add Python to PATH"** kutusunu işaretleyin.

## 2. Klasörü kopyalayın

Tüm klasörü olduğu gibi kopyalayın. **`wheels/` klasörü mutlaka gelmeli** —
offline kurulumun tek kaynağı odur (~161 MB, 79 paket).

> ⚠️ GitHub'dan `git clone` yapmayın: `.gitignore` `wheels/` klasörünü hariç
> tutar, klonda paketler gelmez ve kurulum yapılamaz. **USB / dosya kopyası ile taşıyın.**

## 3. Kurulum

```
setup.bat
```

Sanal ortam oluşturur ve tüm kütüphaneleri `wheels/` klasöründen internetsiz kurar.

## 4. Çalıştırma

```
PCI_RSI_Planner_v3\run_v3_offline.bat
```

Bu dosya `PCI_OFFLINE=1` ile başlatır ve açılışta erişim adreslerini yazar.

## 5. Kurum içi erişim (güvenlik duvarı)

Bir kez, **yönetici** komut isteminde:

```
netsh advfirewall firewall add rule name="PCI RSI Planner" dir=in action=allow protocol=TCP localport=8502
```

Sonra ağdaki herkes tarayıcıdan erişir:

```
http://<sunucu-makine-ip>:8502
```

Sadece bu makineden erişim isterseniz [.streamlit/config.toml](PCI_RSI_Planner_v3/.streamlit/config.toml)
içinde `address = "127.0.0.1"` yapın.

---

## Kapalı ağda ne çalışır, ne çalışmaz

| Özellik | Durum |
|---|---|
| Veri yükleme, analiz, çakışma tespiti | ✅ |
| PCI / RSI planlama (Simulated Annealing) | ✅ |
| **Plotly grafikleri** | ✅ Streamlit paketine gömülü, CDN kullanmaz |
| Detaylı raporlar, Excel çıktısı | ✅ |
| Nokia OSS XML çıktısı (NR + LTE) | ✅ |
| Öneriler, tarama, yeni hücre ekleme | ✅ |
| **🗺️ Harita sekmesi** | ❌ **Devre dışı** |

**Harita neden çalışmaz:** Folium haritası, çizim kütüphanesi Leaflet dahil
**15 dış kaynağı** internetten çeker (`leaflet.js`, `markercluster`,
`awesome-markers`, `bootstrap`, `jquery`, `fontawesome` ve
`tile.openstreetmap.org` altlık karoları). Bunlara ulaşılamadığında harita
"bozuk" görünmez — **bomboş beyaz bir kutu** olur, sektör poligonları ve
komşuluk hatları dahil hiçbir şey çizilmez. Bu sessiz başarısızlık yerine
`run_v3_offline.bat` sekmeyi kapatıp durumu açıkça yazar.

Haritaya ihtiyaç olursa: aynı veriyi internete açık bir makinede `run_v3.bat`
ile açıp haritayı oradan üretin, "Haritayı HTML Olarak İndir" ile alın.

---

## Doğrulama (bu paket üzerinde yapıldı)

Temiz bir sanal ortam yalnızca `wheels/` klasöründen kurularak test edildi:

```
Python 3.12.7 — 51 paket offline kuruldu, hata yok

test_engine.py           GEÇTİ
test_nr_pci.py           GEÇTİ
test_carrier_scope.py    GEÇTİ
test_3gpp_tables.py      GEÇTİ
test_oss_export.py       GEÇTİ

Uygulama açılışı (PCI_OFFLINE=1) : 8 sekme, hata yok, harita uyarısı görünüyor
Uygulama açılışı (normal mod)    : 8 sekme, hata yok, harita çalışır durumda
```

---

## Sorun giderme

**`ERROR: Could not find a version that satisfies the requirement pandas`**
Hedef makinedeki Python sürümü 3.12 veya 3.14 değil. `python --version` ile
bakın, uygun sürümü kurun.

**Ağdaki diğer makineler erişemiyor**
Güvenlik duvarı kuralı eklenmemiş (adım 5) veya `config.toml` içinde
`address` `127.0.0.1` olarak ayarlanmış.

**Açılış çok yavaş**
[.streamlit/config.toml](PCI_RSI_Planner_v3/.streamlit/config.toml) dosyasının
uygulama klasöründe olduğundan emin olun; `gatherUsageStats = false` olmazsa
Streamlit her açılışta ulaşamayacağı telemetri sunucusunu bekler.
