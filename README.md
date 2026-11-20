# Karşılaştırmalı Duygu Analizi (Sentiment Analysis)

Kural tabanlı (**VADER**), derin öğrenme tabanlı (**BERT**) ve büyük dil modeli tabanlı (**LLM**) duygu analizi yöntemlerinin İngilizce ve Türkçe metinler üzerinde karşılaştırılması.

**Teslim:** 25 Kasım 2026
**Grup:** Ceren · Elisar · Nazlıcan

---

## Veri setleri

| Dil | Veri seti | Kaynak | Sınıflar |
|---|---|---|---|
| İngilizce | Twitter US Airline Sentiment | Kaggle | Pozitif / Negatif / Nötr |
| Türkçe | winvoker Turkish Sentiment Analysis | Hugging Face | Pozitif / Negatif / Nötr |

Her iki dil için test kümesi **sınıf başına 1.000 örnek** (toplam 3.000) içerir ve `seed = 42` ile ayrılmıştır.
Metinler küçük harfe çevrilmez ve noktalama silinmez. Tüm yöntemler aynı test kümesi üzerinde değerlendirilir.

## Yöntemler

| Yöntem | İngilizce | Türkçe |
|---|---|---|
| **VADER** (kural tabanlı) | VADER, ham metin | Opus-MT ile TR→EN çeviri + VADER |
| **BERT** (derin öğrenme) | `cardiffnlp/twitter-roberta-base-sentiment-latest` (hazır fine-tuned) | `dbmdz/bert-base-turkish-cased` (BERTurk), fine-tune: 2 epoch, lr 2e-5, batch 16, max_len 128, fp16 |
| **LLM** (zero-shot) | `Qwen2.5-7B-Instruct` (4-bit NF4, yerel) | aynı model, aynı prompt |

**VADER ek deneyi:** Çevirinin katkısını ölçmek için Türkçe metin çevrilmeden doğrudan VADER'a da verilmiştir (`preds_vader_tr_ham.csv`).

**LLM ayrıntıları:** Zero-shot, iki dilde aynı İngilizce talimat; greedy decoding (temperature 0'a eşdeğer, deterministik); cevap tek kelime (*Positive / Negative / Neutral*) olarak istenir. Geçersiz cevapta bir kez daha sorulur, yine geçersizse `Geçersiz` olarak işaretlenir.
Başlangıçta Gemini API denendi; ücretsiz katmanın günlük istek kotası (20 istek/gün) yetersiz kaldığı için tekrarlanabilir ve kota sınırı olmayan yerel bir açık kaynak modele geçildi.

## Klasör yapısı

```
duygu-analizi/
├── data/                         # test_en.csv, test_tr.csv, train_tr.csv  (id,text,label)
├── notebooks/
│   ├── 01_veri_hazirlama.ipynb   # veri setlerinin indirilmesi, temizlenmesi, test/train ayrımı
│   ├── 02_vader.ipynb            # VADER (EN doğrudan, TR çeviri + VADER, TR ham)
│   ├── 03_bert.ipynb             # BERT (EN RoBERTa, TR BERTurk fine-tune)
│   ├── 04_llm_local.ipynb        # LLM (Qwen2.5-7B-Instruct, zero-shot)
│   └── 05_karsilastirma.ipynb    # yöntemlerin karşılaştırma tablo ve grafikleri
├── src/
│   └── evaluate.py               # ortak değerlendirme (accuracy, macro-F1, sınıf bazlı hata, confusion matrix)
└── results/
    ├── preds_<model>_<dil>.csv   # id,text,true,pred  (model: vader / bert / llm, dil: en / tr)
    ├── preds_vader_tr_ham.csv    # çevirisiz Türkçe metinle VADER (ek deney)
    ├── ceviri_tr_en.csv          # Opus-MT ile TR→EN çeviriler
    ├── llm_raw_local_<dil>.jsonl # LLM'in ham cevapları
    ├── *_pilot.*                 # LLM pilot çalıştırması (sınıf başına 20 örnek)
    ├── berturk_training_info.json# BERTurk eğitim bilgileri
    └── figures/                  # confusion matrix görselleri
```

## Çalıştırma

Tüm notebook'lar **Google Colab** üzerinde, **T4 GPU** çalışma zamanıyla çalıştırılmıştır.

### Veri kurulumu

Hazır test/eğitim kümeleri `data/` klasöründe bulunur (`test_en.csv`, `test_tr.csv`, `train_tr.csv`); 02–05 notebook'ları için yeniden üretmeye gerek yoktur. `01_veri_hazirlama.ipynb` baştan çalıştırılmak istenirse:

- **İngilizce:** [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) veri setinden `Tweets.csv` dosyasını Kaggle'dan indirin ve Google Drive'da `MyDrive/DuyguAnalizi/ham_veri/Tweets.csv` konumuna koyun.
- **Türkçe:** [winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) notebook içinde Hugging Face `datasets` kütüphanesiyle otomatik indirilir.

`01` ve `02` notebook'ları dosyaları Drive'daki `MyDrive/DuyguAnalizi/` klasöründen okur (`KLASOR` değişkeni). `01` çalıştırılmayacaksa repodaki `data/` ve `src/` klasörlerini bu klasöre kopyalayın ya da `KLASOR` değişkenini repo klasörünü gösterecek şekilde değiştirin.

### Adımlar

1. Notebook'u Colab'da açın: *Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU*.
2. Repo özel olduğu için Colab *Secrets* bölümüne `GITHUB_TOKEN` ekleyin ve *Notebook access*'i açın.
3. Notebook'ları sırayla çalıştırın: `01_veri_hazirlama` → `02_vader` → `03_bert` → `04_llm_local` → `05_karsilastirma`.
4. Her yöntem tahminlerini `results/preds_<model>_<dil>.csv` dosyasına yazar. Ölçüm için:

```bash
python src/evaluate.py results/preds_llm_en.csv
```

`04_llm_local.ipynb` notları:
- İlk adımda `PILOT_PER_CLASS = 20` ile kısa bir pilot çalışır; tüm veri için `None` yapılır.
- Kaldığı yerden devam eder: ham cevaplar `results/llm_raw_local_<dil>.jsonl` dosyasına her 16 örnekte bir yazılır.
- Tüm veri T4 üzerinde yaklaşık 1 saat 15 dakika sürer (EN ~33 dk, TR ~42 dk).

## Sonuçlar

Test kümesi: her dil için 3.000 örnek (sınıf başına 1.000).

| Yöntem | Dil | Accuracy | Macro-F1 | Hatalı: Pozitif | Hatalı: Negatif | Hatalı: Nötr | Geçersiz |
|---|---|---|---|---|---|---|---|
| VADER | EN | 0,558 | 0,537 | 91 | 567 | 668 | – |
| VADER (çeviri + VADER) | TR | 0,679 | 0,676 | 178 | 455 | 331 | – |
| BERT | EN | 0,780 | 0,775 | 129 | 134 | 396 | – |
| BERT | TR | 0,947 | 0,947 | 47 | 111 | 0 | – |
| LLM | EN | 0,798 | 0,793 | 121 | 106 | 378 | 2 |
| LLM | TR | 0,781 | 0,786 | 200 | 249 | 207 | 23 |
| *VADER (ham, çevirisiz) — ek deney* | *TR* | *0,364* | *0,232* | *932* | *970* | *6* | *–* |

*Hatalı tahmin sayıları gerçek sınıfa göredir (o sınıfa ait olup yanlış tahmin edilen örnek sayısı). Geçersiz cevaplar hatalı sayılır. VADER ve BERT her örnek için geçerli bir etiket üretir.*

Çevirisiz VADER, sözlüğü İngilizce olduğu için Türkçe kelimelerin çoğunu tanımaz ve 3.000 cümlenin 2.837'sini Nötr olarak etiketler. TR→EN çeviri accuracy'yi 0,364'ten 0,679'a çıkarır (yaklaşık +31 puan).

Ayrıntılı karşılaştırma ve yorumlar proje raporunda yer almaktadır.

## Gereksinimler

`transformers`, `torch`, `bitsandbytes`, `accelerate`, `pandas`, `scikit-learn`, `vaderSentiment`, `matplotlib`
(Colab'da çoğu hazır gelir; eksik olanlar notebook içinde `pip install` ile kurulur.)
