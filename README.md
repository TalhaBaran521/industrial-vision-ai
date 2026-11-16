# Industrial Vision AI

Endüstriyel ürün görüntülerindeki yüzey kusurlarını ve normalden sapmaları
incelemeyi hedefleyen bir araştırma ve portföy projesi. Mevcut Phase 0 sürümü
Python paketini, komut satırı girişini, uygulama ayarlarını ve test altyapısını
içerir.

## Kurulum

Python 3.11 ile proje kökünde PowerShell üzerinden çalıştırın. Komutlar sanal
ortamın yürütülebilir dosyalarını doğrudan kullanır.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`py` kurulu değilse, aynı ortamı uv ile oluşturabilirsiniz.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"
```

## Çalıştırma ve test

```powershell
.\.venv\Scripts\industrial-vision.exe --help
.\.venv\Scripts\industrial-vision.exe --version
.\.venv\Scripts\industrial-vision.exe status
.\.venv\Scripts\industrial-vision.exe --config .\configs\application.toml status
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

`status` ile paketin komut satırından çalıştığını ve etkin log seviyesini
kontrol edebilirsiniz. Varsayılan seviye `INFO`'dur. Ayrıntılı log için
`configs/application.toml` dosyasını `configs/application.local.toml` adıyla
kopyalayın, kopyadaki `log_level` değerini `"DEBUG"` yapın ve dosyayı
`--config` ile belirtin. Loglar standart hata akışına yazılır.

## Mimari ve bağımlılıklar

Paket `src/industrial_vision` altında yer alır. Phase 0 çalışma zamanında Python
standart kütüphanesini kullanır. Paketleme için setuptools, geliştirme için
`dev` ek bağımlılıkları olan pytest ve Ruff kullanılır. Katmanlar ve veri akışı
[mimari belgesinde](docs/architecture.md) açıklanmıştır.

## Geliştirme sırası

Phase 1'de görüntü doğrulama ve yükleme, ön işleme ve Streamlit önizlemesi
eklenecek. Phase 2'de sentetik örneklerle referans tabanlı görsel fark analizi
yapılacak. Gerçek veri kümesi, PatchCore ve deneysel değerlendirme sonraki
aşamaların kapsamındadır.

## Phase 1 görüntü ön işleme

Kurulumdan sonra `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1` komutunu çalıştırın. PNG/JPEG yükleyin, kenar çubuğundan işlem ve parametreleri seçin. Çıktıyı PNG olarak indirebilirsiniz. Görüntüler RGB uint8 olarak tutulur. EXIF yönü uygulanır, şeffaf alanlar beyazla birleştirilir. Yükleme sınırı 10 MB, piksel sınırı 20 milyondur ve arayüzden ayarlanabilir. Morphology seçimi önce belirtilen eşikle binary maske oluşturur. Diğer işlemler doğrudan orijinal görüntüye uygulanır. Bu aşama ön işleme araçlarını içerir.

## Phase 2 — Görsel fark analizi

PowerShell üzerinden `.\.venv\Scripts\industrial-vision.exe demo --seed 42 --output outputs/inspections` komutuyla sentetik örnek üretip analiz edin. `--threshold 25` ve `--min-area 20` parametreleri ayarlanabilir. Her çalıştırma benzersiz bir dizinde referans, kusurlu görüntü, gerçek maske, fark haritası, filtrelenmiş maske, kutulu görüntü, overlay ve metadata.json üretir. Gerçek maske yalnızca sentetik örneğin değiştirilen piksellerini gösterir ve detector girdisi olarak kullanılmaz.

Yöntem her pikselde RGB kanallarının mutlak farkının maksimumunu alır. Eşiği kesin olarak aşan pikseller maskeye alınır. Varsayılan 3×3 opening ardından 8-komşulukla bağlı bileşenler bulunur ve 20 pikselden küçük bölgeler çıkarılır. Alan, bileşendeki piksel sayısıdır. Kutular x, y, width, height ile orijinal, EXIF yönü uygulanmış görüntü koordinatlarındadır. Analizde yeniden boyutlandırma yapılmaz. `kernel_size=1` opening işlemini kapatır.

### Sınırlamalar

Bu yöntem görsel fark analizi yapar. Aynı çözünürlük, hizalanmış sahne, aynı kamera açısı ve benzer aydınlatma gerektirir. Kamera hareketi, gölge, pozlama, yansıma ve JPEG artefaktları yanlış pozitif üretebilir. Opening ince çizikleri silebilir; eşik ve alan filtresi küçük veya düşük kontrastlı kusurları kaçırabilir. Otomatik hizalama uygulanmaz. `is_anomalous`, filtrelerden sonra bölge bulunup bulunmadığını belirtir. Anomaly score ve anomaly threshold alanları null tutulur; piksel fark eşiği configuration içinde kaydedilir. Sentetik sonuçlar yazılım doğrulaması içindir, gerçek endüstriyel başarı ölçümü sağlamaz.

İşlem süresi bellek içi analiz ve görselleştirmeyi kapsar. Dosya yükleme ve kayıt bu ölçümün dışındadır. Kayıt hatasında analiz bellekte tutulur ve warnings alanına hata eklenir. CLI hata durumunda sıfırdan farklı çıkış kodu verir.
