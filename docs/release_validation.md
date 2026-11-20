# Phase 8 doğrulama ve performans

20 Kasım 2026 tarihinde Windows ve Python 3.11.15 ortamında çalıştırıldı.

| Kontrol | Gerçek sonuç |
| --- | --- |
| ML ortamı pytest | 58 passed, 1 warning; 134.10 saniye |
| Ayrı temel ortam, dev bağımlılıkları | 50 passed, 4 skipped; 27.90 saniye |
| Ruff check | Geçti |
| Ruff format check | 45 Python dosyası biçimlendirilmiş |
| uv pip check | 99 paket kontrol edildi |
| git diff --check | Geçti |
| Takip edilen data/models/outputs | Dosya bulunmadı |
| Sentetik MP4 yeniden decode | 80 frame, 10 FPS, 8 saniye |
| Tarayıcı kontrolü | Experiment Results sayfası açıldı ve screenshot kaydedildi |

Temel ortam outputs/release-check altında uv ile sıfırdan oluşturuldu ve yerel paket önbelleğinden editable kurulum yapıldı. PyTorch ve Anomalib içermeyen bu ortamda ML testleri uygun şekilde atlandı. Bu kontrol temiz ağ indirmesi veya farklı Windows bilgisayarı doğrulaması değildir. Anomalib coreset API deprecation uyarısı sürüyor.

## Ölçülen performans

Phase 7 kayıtlı video denemesi RTX 4070 Laptop GPU ve CUDA üzerinde 12 frame işledi. Kaynak 320×320, aynı bottle görüntüsünün tekrarından oluşturulan MJPEG videodur. Üç warm-up sonrası işlem FPS 18.70, ortalama gecikme 52.10 ms, p50 49.58 ms, p95 62.36 ms oldu. Bu kısa tek çalışmadır; uzun süreli veya endüstriyel kamera performansı hakkında sonuç vermez. Kamera kaynaklı frame kaybı ölçülmedi. Kaynak rapor outputs/video/e6bd85bc789b489494d5335d748659be.json dosyasındadır.

## Portföy kapsamı

README İngilizce özet, PowerShell kurulum/çalıştırma komutları, mimari, veri hazırlama ve model kullanımını içerir. docs/experiments.md gerçek değerlendirmeyi, docs/model_card.md model kapsamını, docs/dependencies.md bağımlılık envanterini açıklar. docs/media altında sentetik sonuç görseli, kısa MP4 demo ve gerçek arayüz screenshot bulunur. Sentetik demo yazılım akışını gösterir.

## Açık sınırlamalar

Fiziksel webcam ve kamera penceresi etkileşimi doğrulanmadı. Tarayıcı dosya yükleme/indirme tıklamaları uçtan uca doğrulanmadı; AppTest ve uygulama servis testleri mevcuttur. Proje kaynak kodunun lisansı henüz seçilmedi. GitHub yayını ve commit bu aşamada yapılmadı. Çalışma alanında önceki aşamalara ait değişiklikler de vardır. Model/veri dosyaları Git kapsamı dışındadır; yeni geliştirici resmî veriyi edinip modeli CLI ile üretmelidir.
