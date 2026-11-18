# Phase 5 deney raporu

18 Kasım 2026 çalışması, run 7b2d85a0a51e40169984e76c1560db7e. CUDA üzerinde bottle kategorisinin 83 test görüntüsü ve 67.230.000 pikseli değerlendirildi. Model hash'i ve tüm ayarlar çıktının report.json dosyasındadır. Görüntü eşiği 10.137834167480468 olarak korundu. Piksel eşiği 41 normal validation haritasından 8.825857162475586 hesaplandı. Test verisi eşik hesabına katılmadı.

| Metrik | Görüntü | Piksel |
| --- | --- | --- |
| AUROC | 0.979365 | 0.966891 |
| Average Precision | 0.994540 | 0.540621 |
| Precision | 0.968254 | 0.298239 |
| Recall | 0.968254 | 0.979925 |
| F1 / Dice | 0.968254 | 0.457300 |

Piksel IoU 0.296428. Görüntü confusion matrix TN=18, FP=2, FN=2, TP=61. Toplam değerlendirme süresi 149.09 saniye; tarama, validation, tahmin ve metrik hesabını kapsar.

## Hata analizi

Normal good/007.png ve good/008.png görüntüleri yanlış pozitif oldu. contamination/003.png ve contamination/019.png kusurları görüntü eşiğinin altında kaldı. predictions.csv tüm skorları içerir. TP/FP/TN/FN örneklerinin orijinal, ground truth ve tahmin maskeleri rapor klasörüne kaydedildi.

Piksel recall yaklaşık %98 iken precision yaklaşık %30. Bu eşikte geniş yanlış pozitif bölgeler oluşuyor; 8.961.934 normal piksel kusurlu işaretlendi. Dolayısıyla görüntü sınıflandırma sonuçları, kusurun sınırlarını hassas belirleme başarısı olarak yorumlanmamalıdır. Düşük model çözünürlüğü ve normal örneklerden seçilen piksel eşiği bu hata türleri açısından sonraki bağımsız deneylerde incelenebilir. Bu test sonuçlarına bakılarak eşikler değiştirilmedi.

51 test geçti, Ruff temiz. Anomalib coreset çağrısına ait bir deprecation uyarısı sürüyor. Metrikler yalnızca bu kategori, sabit model ve bölme için geçerlidir.
