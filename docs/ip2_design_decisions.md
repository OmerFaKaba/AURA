# İP2 Tasarım Kararları

## Deney protokolü

- Veri: `data/processed/milan_hourly.parquet`, 504 saat × 400 hücre.
- Zamansal bölme: ilk %70 eğitim (352), sonraki %15 doğrulama (76), son %15 test (76).
- Normalizasyon istatistikleri yalnızca eğitim bölümünden hesaplandı.
- Pencereleme her bölüme ayrı uygulandı; bölme sınırlarını geçen örnek oluşturulmadı.
- Mimari karşılaştırması 42, 123 ve 456 seed'leriyle aynı veri ve hiperparametrelerde yapıldı.
- Mimari seçimi test setine göre değil, ortalama doğrulama MAE'sine göre yapıldı.
- Test metrikleri ters normalizasyon sonrasında özgün veri ölçeğinde hesaplandı.

## Karar tablosu

| Karar | Denenen | Seçilen | Gerekçe |
|---|---|---|---|
| Model mimarisi | MLP, 1D-CNN, LSTM; her biri 3 seed | LSTM | En düşük ortalama doğrulama MAE'si: 220.42 ± 1.75 |
| Pencere boyutu | 24 | 24 saat | Günlük aktivite döngüsünü tam kapsıyor; 12/48 ablasyonu bu İP kapsamında çalıştırılmadı |
| Tahmin ufku | 1 | 1 saat | Bir sonraki saat tahmini merkezi baseline sözleşmesiyle uyumlu |
| Optimizer | Adam | Adam | Üç mimaride de aynı optimizer kullanılarak adil karşılaştırma sağlandı |
| Öğrenme oranı | 1e-3 | 1e-3 | Tüm mimarilerde sonlu ve azalan loss ile kararlı eğitim verdi |
| Normalizasyon | Hücre bazlı standartlaştırma | Standartlaştırma | Yoğunluk farklarını dengeler ve ters dönüşüm için ortalama/std saklanabilir |
| Batch boyutu | 64 | 64 | GPU belleğine rahatça sığdı ve tüm modellerde sabit tutuldu |
| Erken durdurma | Patience 10 | Patience 10 | Tüm koşular en iyi epoch'tan 10 epoch sonra durdu; gereksiz eğitimi önledi |
| Temsilî checkpoint seed'i | 42, 123, 456 karşılaştırıldı | 123 | Seçilen LSTM mimarisinde en düşük doğrulama MAE'sine sahip koşu |

## Model karşılaştırma sonucu

| Model | Validation MAE (ort ± std) | Test MAE (ort ± std) | Parametre | Ortalama süre |
|---|---:|---:|---:|---:|
| LSTM | 220.42 ± 1.75 | **215.21 ± 3.53** | 17,217 | 144.54 sn |
| MLP | 237.35 ± 3.19 | 221.33 ± 1.00 | 1,665 | 142.14 sn |
| 1D-CNN | 240.45 ± 2.52 | 228.76 ± 3.15 | 11,521 | 110.93 sn |

LSTM, CNN'den daha büyük olsa da 17,217 parametre federated-learning iletişimi açısından hâlâ küçük bir modeldir. İP3'te bütün istemciler aynı `LSTMModel` sınıfını kullanacaktır.

## Baseline raporlama kararı

Bilimsel ana sonuç, seed seçimine bağlılığı göstermek için üç seed'in ortalama ve standart sapmasıyla raporlanır:

> Merkezi LSTM baseline test MAE: **215.21 ± 3.53**

Dağıtılabilir model dosyası için test sonucuna bakarak seed seçilmedi. En iyi doğrulama MAE'sine sahip seed 123 yeniden eğitilip `models/baseline_lstm.pt` olarak kaydedilir.

Temsilî seed 123 checkpoint sonuçları:

| Metrik | Değer |
|---|---:|
| Test MAE | 219.29 |
| Test RMSE | 356.08 |
| Test MAPE | %7.94 |
| En iyi epoch | 10 |
| Early stopping tamamlanma epoch'u | 20 |

Loss grafiğinde eğitim kaybı düşmeye devam ederken doğrulama kaybının epoch 10 sonrasında iyileşmemesi overfitting başlangıcını gösterir. Early stopping bu nedenle epoch 10 ağırlıklarını geri yükler.

## Sınırlılıklar ve sonraki deneyler

- 12 ve 48 saatlik pencere ablasyonu henüz uygulanmadı; 24 saat seçimi alan bilgisine dayanıyor.
- SGD ve alternatif öğrenme oranları karşılaştırılmadı; mevcut sonuçlar yalnızca mimari karşılaştırmasını izole ediyor.
- MAPE hesaplanırken gerçek değeri sıfıra çok yakın örnekler dışlanıyor; ana karşılaştırma metriği MAE'dir.
- İP3'te normalizasyon merkezi değil, istemci-yerel istatistiklerle uygulanacaktır.
