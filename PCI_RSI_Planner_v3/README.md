# PCI / RSI Planner v3

3GPP TS 36.211 / 38.211 / 38.331'e göre 4G ve 5G PCI ve PRACH RSI planlama aracı.

Üst klasördeki v2 dokunulmadan duruyor (`v2.0` etiketi) — ikisi aynı anda
çalışabilir.

```
run_v3.bat        ->  http://localhost:8502
..\run.bat        ->  http://localhost:8501   (v2, donmuş)
```

## v2'den farkı

| | |
|---|---|
| **Teknoloji** | Arayüz seçimi tek otorite; PCI aralığı (LTE 0-503 / NR 0-1007) her planlayıcı çıktısında doğrulanır |
| **Çakışma kapsamı** | Taşıyıcı bazlı — farklı ARFCN'deki hücreler arasında çakışma sayılmaz. Samsun'da bulguların %62'si bu yüzden sahteymiş |
| **Planlama stratejisi** | Sektör bazlı (tüm taşıyıcılar tek PCI) veya taşıyıcı bazlı — seçilebilir |
| **Trafik ağırlığı** | Çakışma cezaları HO attempt sayısıyla ölçeklenir; kaçınılmaz çakışma sessiz ilişkilere kayar |
| **PRACH tabloları** | Spec metninden çıkarılmış ve yapısal olarak doğrulanmış (`prach_tables.py`) |
| **Kısıtlı küme** | Yüksek hız (typeA/typeB) kök sayımı TS 36.211 §5.7.2 ile kesin |
| **Hücre menzili** | Ncs penceresi `1/Δf_RA` — LTE format 2/3 ve NR format 3 artık doğru |
| **Komşuluk** | Mekânsal ızgara `cos(enlem)` ile ölçekli; kaba kuvvetle doğrulanmış, 0 kaçan |
| **OSS çıktısı** | Nokia RAML 2.0 — LTE (`LNCEL` + `LNCEL_FDD`) ve NR (`NRCELL`) |

## Veri şeması

Zorunlu: `cell_id`, `latitude`, `longitude`, `pci`

Önemli opsiyoneller:

| Sütun | Ne işe yarar |
|---|---|
| `earfcn` / `arfcn` | **Taşıyıcı anahtarı.** Band aynı banttaki iki taşıyıcıyı ayıramaz |
| `dist_name` | OSS XML çıktısı için. **Yoksa XML üretilmez** — türetilmez |
| `msg1_scs_khz` | NR kısa preamble hücre menzili (15/30/60/120) |
| `duplex` | FDD/TDD — boşsa banttan türetilir |
| `high_speed` | `unrestricted` / `typeA` / `typeB` |
| `rsi`, `prach_config_index`, `zero_correlation_zone`, `cell_range` | PRACH / RSI planlaması |

Örnek dosyayı arayüzden indirebilirsiniz; "Format Bilgisi" sayfası her sütunu
açıklar.

## Testler

```
python test_engine.py          devralınan regresyon takımı
python test_nr_pci.py          teknoloji otoritesi ve PCI aralıkları
python test_carrier_scope.py   taşıyıcı kapsamı, planlama modu, arayüz nöbetçileri
python test_3gpp_tables.py     3GPP tablolarının bağımsız doğrulaması
python test_oss_export.py      Nokia RAML — gerçek export'larla round-trip
```

`test_3gpp_tables.py` tabloları spec'ten bağımsız olarak yeniden uygular ve
motoru ona karşı sınar. Tabloyu hafızadan yazmak bir kez hem kodu hem testi
aynı şekilde yanlış yapmıştı; o yüzden referans uygulama ayrı.

## v2 ile karşılaştırma

```
python compare_v2_v3.py --data <hucreler.xlsx> --tech LTE
```

İki motoru ayrı süreçlerde aynı veriyle koşturur; özet, komşuluk grafiği,
hücre bazlı PRACH parametreleri ve plan çıktılarını diff'ler. Bir planı devreye
almadan önce farkların hepsinin kasıtlı düzeltmelere karşılık geldiğini
doğrulamak için.

## Bilinen sınırlar

- **Yarıçap 2-3 km tutulmalı.** HO listesi yüklüyken 20 km, birbirini hiç
  görmeyen hücreler arasında on binlerce uydurma ilişki ekler; hem skor hem
  plan onlara göre şekillenir.
- **NR mod-30 varsayılan kapalı.** `hoppingId` / `nPUSCH-Identity` konfigüre
  edilmemişse anlamlı — konfigürasyonu kontrol edip açın.
- **`parse_band_info` operatöre özel** ve tanımadığı öneki "Bilinmeyen"e düşürür;
  bu yalnızca haritadaki bant renklendirmesini etkiler, kapsam ARFCN'den gelir.
- **v3 uçtan uca yalnızca NR verisiyle sürüldü.** LTE yolu testlerde geçiyor
  ama gerçek LTE verisiyle bir kez `compare_v2_v3.py` koşturmak gerekir.
