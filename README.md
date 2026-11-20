# Industrial Vision AI

Endüstriyel görüntüler için Python tabanlı araştırma ve portföy prototipi. Görüntü ön işleme, referansla görsel fark analizi, MVTec AD keşfi, PatchCore eğitim/tahmini, bilimsel değerlendirme ve yerel video analizi içerir.

## English overview

Industrial Vision AI is a modular Python research prototype for industrial visual inspection. It combines OpenCV preprocessing, reference-based difference analysis, a Streamlit interface, and Anomalib PatchCore trained on normal MVTec AD bottle images. Training, validation and test data are separated. Reports include image-level and pixel-level metrics, failure examples and measured video latency. The current model is category-specific and requires further validation for industrial deployment.

## Hızlı başlangıç

Python 3.11 sanal ortamını aşağıdaki komutlarla kurduktan sonra Streamlit uygulamasını açın.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
.\.venv\Scripts\industrial-vision.exe demo --seed 42
```

Temel kurulum ön işleme, Dataset Explorer ve sentetik demo için yeterlidir. PatchCore ekranı için ML bağımlılıklarını kurup bottle modelini eğitim komutuyla oluşturun. Model ve veri dosyaları yerel olarak hazırlanır.

![Sentetik görsel fark analizi](docs/media/classical-inspection.png)

[8 saniyelik sentetik demo](docs/media/classical-demo.mp4) · [Model Card](docs/model_card.md) · [Gerçek deney sonuçları](docs/experiments.md) · [Doğrulama ve performans](docs/release_validation.md) · [Bağımlılıklar](docs/dependencies.md)

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

Paket `src/industrial_vision` altında yer alır. Temel çalışma zamanı NumPy, Pillow, OpenCV ve Streamlit kullanır. Paketleme için setuptools, geliştirme için
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

## Phase 3 — MVTec AD Dataset Explorer

1. [Resmî MVTec AD sayfasından](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) indirme formunu kullanarak bottle kategorisini indirin.
2. Arşivi yerel olarak açın. Proje köküne göre `data/mvtec_ad/bottle/train/good`, `test/good`, `test/<defect_type>` ve `ground_truth/<defect_type>` dizinleri oluşmalıdır. Örneğin `test/broken_large/000.png` için maske `ground_truth/broken_large/000_mask.png` olur.
3. `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1` komutunu çalıştırın ve kenar çubuğundan Dataset Explorer sayfasını açın.
4. Veri kökünü ve bottle kategorisini seçip Scan and validate düğmesine basın. İstatistikler doğrulamadan geçen örnekleri sayar. Hatalı veya eksik örnekler ayrı hata tablosunda görünür. Split ve örnek seçerek görüntü ile ground truth maskesini inceleyin. Dosyalar değiştiğinde tekrar tarayın.

Veri otomatik indirilmez. `data/` dizini .gitignore kapsamındadır. Veri lisansı CC BY-NC-SA 4.0 olup ticari kullanımı kısıtlar; kullanımda resmî lisans ve atıf koşullarını takip edin. Repository testleri geçici dizinlerde küçük sentetik PNG fixture dosyaları üretir.

Loader orijinal train/test ayrımını korur. Eğitim yalnızca train/good örneklerinden oluşur. Anormal test görüntüsü kendi kusur türü ve dosya kökü üzerinden maske ile eşleştirilir. Normal örnekler için sıfır maske oluşturulur. Eksik, bozuk, farklı boyutlu, binary olmayan ve boş anomali maskeleri raporlanır. Eşleşmeyen maskeler ve beklenmeyen eğitim grupları da raporlanır. Tarama her görüntüyü decode ettiği için büyük kategorilerde zaman alabilir. Dosya başına mevcut 10 MB ve 20 milyon piksel sınırları uygulanır.

Validation ayrımı bu aşamada oluşturulmaz. Model aşamasında yalnızca eğitimdeki normal örneklerden ayrı ve tekrarlanabilir validation bölümü seçilmelidir. Test görüntüleri ve ground truth eğitim ya da eşik seçimi için kullanılmamalıdır. Explorer'daki maskeler veri kümesinin anotasyonlarıdır. Gerçek veri bulunmayan ortamda fixture testleri loader davranışını doğrular; gerçek kategori üzerinde bütünlük kontrolü ayrıca çalıştırılmalıdır.

## Phase 4 — PatchCore

ML bağımlılıkları isteğe bağlı `ml` grubundadır. Windows PowerShell kurulumunda önce resmî CUDA PyTorch paketlerini, sonra proje ek bağımlılıklarını kurun. CPU için ilk komuttaki cu128 yerine cpu kullanın. İlk eğitim timm/resnet18.a1_in1k ImageNet ağırlıklarını Hugging Face üzerinden indirir.

```powershell
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[dev,ml]"
$env:HF_HOME = Join-Path $PWD '.uv-cache/huggingface'
$env:TORCH_HOME = Join-Path $PWD '.uv-cache/torch'
.venv/Scripts/python.exe -m industrial_vision.ml_cli device
.venv/Scripts/python.exe -m industrial_vision.ml_cli train --data data/mvtec_ad --device auto --seed 42 --image-size 128
.venv/Scripts/python.exe -m industrial_vision.ml_cli predict --model models/patchcore/MODEL_ID --image data/mvtec_ad/bottle/test/good/000.png --device auto
```

Eğitim çıktısındaki MODEL_ID dizinini tahmin komutunda kullanın. `--device cpu` CPU çalışmasını seçer; auto ve cuda seçimleri CUDA bulunmadığında CPU'ya düşer. Python'daki torch.cuda.is_available kontrolü esas alınır.

Anomalib 2.2.0 PatchcoreModel API'si kullanılır. Önceden eğitilmiş ResNet18 layer2/layer3 özellikleri, 128×128 tam görüntü resize, ImageNet normalizasyonu, tek görüntülük batch, %1 coreset ve bir komşu kullanılır. Bu hafif başlangıç yapılandırması makaledeki varsayılan deneyle aynı değildir. PatchCore normal özelliklerden bellek bankası oluşturur. Normal train/good görüntülerinin %20'si sabit seed ile ayrılır; eğitim ve validation manifestleri metadata.json'a kaydedilir. Test görüntüleri veya maskeler eğitim ve eşik hesabına girmez.

Görüntü eşiği normal validation skorlarının %99 yüzdeliğidir. Bu eşik sınırlı normal örneklerden tahmin edilir ve anormal örnekleri yakalama başarısını garanti etmez. Ham skor bir mesafe ölçüsüdür, olasılık değildir. Piksel eşiği belirlenmez. Tahmin, gerçek anomali haritasını float NPY ve skoru JSON olarak benzersiz outputs/predictions dizinine yazar. Harita orijinal görüntü boyutuna linear interpolasyonla getirilir. Ground truth tahmin girdisi olarak okunmaz.

Model dizini tensor state_dict içeren model.pt ve metadata.json taşır. Yükleme weights_only=True kullanır, backbone ağırlıklarını tekrar indirmez. Yalnızca kendi ürettiğiniz model dizinlerini kullanın. Model yüklenmesi temel uygulama importlarında tetiklenmez. Bellek yetersizliğinde --image-size 64 veya --device cpu kullanılabilir. Model değerlendirme metrikleri Phase 5 kapsamındadır.

## Phase 5 — Model değerlendirme

```powershell
.venv/Scripts/python.exe -m industrial_vision.ml_cli evaluate --data data/mvtec_ad --model models/patchcore/51ad27ecf5134c64abddbdb9cd2ba44e --device auto
```

Komut modelin kayıtlı görüntü eşiğini kullanır. Piksel eşiği, metadata içindeki normal validation görüntülerinin her birindeki %99.5 piksel yüzdeliğinin %99 yüzdeliğinden hesaplanır. Test etiketleri veya test skorları eşik seçiminde kullanılmaz. Bu yöntem normal doğrulama örneklerine bağlıdır ve kusurları yakalama garantisi sağlamaz.

Bütün test görüntüleri için image AUROC, Average Precision, Precision, Recall, F1 ve confusion matrix hesaplanır. Piksel metrikleri orijinal çözünürlükteki tüm test pikselleri birleştirilerek hesaplanır; örnekleme yapılmaz. Pixel AUROC/AP ile belirlenen piksel eşiğinde IoU ve Dice raporlanır. Bu toplu hesap birkaç GB RAM kullanabilir. Görüntü ve piksel görevlerinin metrikleri ayrı raporlanır. Tanımsız değerler JSON'da null, CSV'de N/A olur. Confusion matrix sırası [[TN, FP], [FN, TP]] şeklindedir.

Benzersiz outputs/evaluations/RUN_ID dizininde report.json, metrics.csv, predictions.csv ve her TP/FP/TN/FN grubu için en fazla üç örneğin orijinali, ground truth ve tahmin maskesi saklanır. Örnekler deterministik dosya sırasından seçilir; örneği bulunmayan grup boş listelenir. predictions.csv bütün hatalı örnekleri incelemek için kullanılabilir. Rapor model hash'i, model ayarları, seed, veri manifestleri, sürümler, cihaz, eşikler ve toplam süreyi içerir. Süre tarama, kalibrasyon ve metrik hesabını da içerir; inference benchmark değildir.

## Phase 6 — Kullanıcı arayüzü

`.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1` ile açın. Ana sayfa ön işleme araçlarını ve sayfa bağlantılarını içerir. Image Inspection ekranında yerel models/patchcore içindeki modeli ve cihazı seçip PNG/JPEG yükleyin, Analyze image düğmesine basın. Orijinal, heatmap ve overlay ile skor, eşik, karar ve işlem süresi gösterilir. Overlay PNG, JSON raporu ve ham NPY haritası indirilebilir. Model her oturumda ayrı tutulur; indirilecek görüntüler ortak cache'e yazılmaz.

Heatmap renkleri görüntüye özel 0–maksimum ham mesafe aralığındadır. Farklı görüntülerin renkleri doğrudan karşılaştırılmamalıdır. Görüntü kararı kayıtlı validation eşiğine dayanır. Tek görüntü analizinde piksel eşiği uygulanmaz. Experiment Results sayfası kayıtlı değerlendirmeleri, iki düzeyde metrikleri, confusion matrix ve TP/FP/TN/FN örneklerini gösterir. Buradaki predicted mask değerlendirme raporundaki piksel eşiğini kullanır. JSON ve CSV raporları indirilebilir. Dataset Explorer üzerinden veri ve ground truth incelenebilir.

## Phase 7 — Kamera ve video analizi

```powershell
.venv/Scripts/python.exe -m industrial_vision.ml_cli video --model models/patchcore/51ad27ecf5134c64abddbdb9cd2ba44e --camera 0
.venv/Scripts/python.exe -m industrial_vision.ml_cli video --model models/patchcore/51ad27ecf5134c64abddbdb9cd2ba44e --video C:/videos/inspection.mp4 --headless --max-frames 100
```

Yerel OpenCV penceresinde Q, Esc veya pencereyi kapatma ile çıkın. Ctrl+C de kaynağı serbest bırakır. Kamera seçimi fiziksel kamerayı açar. `--headless` pencere oluşturmaz. `--device cpu` desteklenir. Model bottle verisiyle eğitilmiştir; genel webcam sahneleri için geçerlilik iddiası yoktur.

Frame BGR'den RGB'ye çevrilir ve eğitimle aynı ön işleme uygulanır. Frame dosyaya yazılmadan predict_rgb üzerinden işlenir. İlk frame üzerinde üç warm-up tahmini ölçümden çıkarılır. Ortalama, p50/p95 read-to-result gecikmesi ve uçtan uca işlem FPS değeri outputs/video altında JSON'a kaydedilir. İlk ölçülen frame'in okuma süresi warm-up nedeniyle dışarıdadır. CUDA sonuçları CPU'ya alındığında senkronizasyon tamamlanır. FPS model yüklemesini ve warm-up süresini dışlar; görüntüleme ve normal frame okuma maliyetini içerir. Dosyalar azami işlem hızında okunur, kaynak FPS hızında oynatılmaz.

Uygulama senkron çalışır ve okuduğu frameleri atlamaz. Kamera sürücüsü/buffer kaynaklı kayıp frame sayısı ölçülemediği için null raporlanır. Bu değer sıfır kayıp anlamına gelmez. JSON skorları frame sırasındadır. Harita renk ölçeği frame'e özel ham mesafedir. Video sonu, frame sınırı ve kullanıcı çıkışı raporlanır; kamera açma/okuma ve video çözme hataları açıklayıcı CLI hatası üretir. Her durumda capture kapatılır.

## Portföy ekranı

![Experiment Results ekranı](docs/media/experiment-results.png)

Test sonuçları ve doğrulama sınırları [release raporunda](docs/release_validation.md) bulunur. Proje kaynak kodu lisansı henüz seçilmemiştir.
