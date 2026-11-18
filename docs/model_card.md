# PatchCore bottle model

17 Kasım 2026 tarihinde MVTec AD bottle train/good içindeki 209 görüntü, seed 42 ile 168 eğitim ve 41 normal validation örneğine ayrıldı. Model kimliği 51ad27ecf5134c64abddbdb9cd2ba44e. Test verileri eğitim ve eşik hesabında kullanılmadı.

Anomalib 2.2.0 PatchcoreModel, ImageNet ön eğitimli timm/resnet18.a1_in1k backbone, layer2/layer3, 128×128 resize, ImageNet normalizasyonu, %1 coreset ve bir komşu kullanıldı. Bellek bankası 430×384 boyutundadır. CUDA RTX 4070 Laptop GPU üzerinde ölçülen eğitim ve validation süresi 17.31 saniyedir. Bu tek çalışma ölçümüdür.

Normal validation skorlarının %99 yüzdeliği 10.137834167480468 olarak bulundu. Test/good/000.png üzerinde yeniden yüklenen model skoru 8.738046646118164, anomali kararı false ve harita boyutu 900×900 oldu. Bu örnek bir çalışabilirlik kontrolüdür. Phase 5 test sonuçları docs/experiments.md belgesinde raporlandı.

Model 128×128 çözünürlük nedeniyle küçük kusurları kaçırabilir. Normal validation örnekleri sınırlıdır ve seçilen eşik hata oranını garanti etmez. Skor olasılık değildir. Phase 5 piksel eşiği normal validation haritalarından 8.825857162475586 olarak belirlendi. Araştırma ve portföy prototipidir; güvenlik kritik otomatik kararlar veya doğrulanmamış endüstriyel kullanım için uygun değildir. Veri lisansı MVTec AD CC BY-NC-SA 4.0 koşullarına tabidir.
