# AURA — İP2 Merkezi Baseline

Bu depo, Milano telekom aktivitesinde bir sonraki saati tahmin eden merkezi PyTorch baseline'ını içerir. İP2 model karşılaştırmasında MLP, 1D-CNN ve LSTM üç farklı seed ile değerlendirilmiş; doğrulama MAE'sine göre LSTM seçilmiştir.

## Ana sonuç

Merkezi LSTM test MAE: **215.21 ± 3.53** (3 seed).

Ayrıntılı kararlar: [`docs/ip2_design_decisions.md`](docs/ip2_design_decisions.md)

## Kurulum

Python 3.12 ile:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CUDA kullanımı için PyTorch'un NVIDIA sürümü ayrıca resmi PyTorch kurulum seçicisinden kurulmalıdır. Scriptler CUDA kullanılabiliyorsa otomatik olarak GPU'yu seçer.

## Veri

Harvard Dataverse'ten indirilen ilk 21 günlük dosyalar şu klasörde olmalıdır:

```text
data/raw/sms-call-internet-mi-2013-11-01.txt
...
data/raw/sms-call-internet-mi-2013-11-21.txt
```

Veriyi üretmek için:

```powershell
python scripts/build_dataset.py
```

Beklenen çıktı: `data/processed/milan_hourly.parquet`, şekil `(504, 400)`.

## Model karşılaştırması

Hızlı entegrasyon kontrolü:

```powershell
python scripts/compare_models.py --smoke
```

Tam 3 model × 3 seed karşılaştırması:

```powershell
python scripts/compare_models.py
```

Karşılaştırma çıktıları `results/` altında saklanır.

## Seçilmiş baseline'ı eğitme

Hızlı kontrol:

```powershell
python scripts/train_baseline.py --smoke
```

Tam eğitim, değerlendirme, checkpoint ve grafik üretimi:

```powershell
python scripts/train_baseline.py
```

Komut aşağıdakileri üretir:

- `models/baseline_lstm.pt`
- `models/baseline_scaler.parquet`
- `results/baseline_metrics.json`
- `results/baseline_loss_history.csv`
- `results/baseline_test_predictions.parquet`
- `results/baseline_per_cell_mae.csv`
- `figures/baseline_loss_curve.png`
- `figures/baseline_prediction_vs_actual.png`
- `figures/baseline_per_cell_mae.png`

Ham veri, işlenmiş veri ve `.venv` Git'e dahil edilmez.
