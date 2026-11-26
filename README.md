# Japonca Karakter Sesi (TTS)

Japonca metni doğal seslendiren, farklı karakter sesleri üreten ve MP3 çıktısı veren
sentetik ses sistemi. Style-Bert-VITS2 (JP-Extra) modeli, JVS + JVNV korpuslarıyla
fine-tune edildi; cümle bazlı duygu stili ve soru tonlaması var. Üzerine hazır cümlelerle
çalışan statik bir sohbet sitesi (`web/`) kuruldu.

**Sürüm: v2 (2026-09-18).** Model `models/finetuned/jvs_jvnv16/`, sitede 4 ses
(Yuki, Hana, Kenta, Takeshi). v1 (JVS-only) `releases/v1/` altında dondurulmuş.

## Klasör Yapısı

```text
configs/         paths.yaml (yollar + prozodi), characters.yaml (karakter preset'leri), lexicon.yaml (okunuş düzeltmeleri)
data/raw/        jvs/ (JVS Corpus, 100 konuşmacı), jvnv/ (JVNV, 4 konuşmacı × 6 duygu)   [git'e konmaz]
metadata/        jvs_manifest*.csv (JVS tarama/filtre sonuçları)
models/finetuned/jvs_jvnv16/   üretim modeli: config.json + *.safetensors + style_vectors.npy   [git'e konmaz]
scripts/         adımlar (aşağıda)
src/japonca_tts/ mantık: config, text_frontend, characters, synth, prosody, export, qc, chat (Ollama)
web/             sohbet sitesi (index.html + dialog.json + manifest.js + audio/); canlı: scripts/serve_chat.py
releases/        v1/ (sayfa + 240 klip + model + QC), v2/ (konuşma kayıtları + QC)
outputs/         wav/, mp3/, logs/ (çalışma çıktıları)                                  [git'e konmaz]
third_party/Style-Bert-VITS2/   eğitim/inference reposu; Data/jvs_jvnv16/models/ = son checkpoint (devam eğitimi için)  [git'e konmaz]
tests/           birim testleri + test/demo/stres cümle setleri
```

Üretim çıktı düzeni (`run_pipeline.py` / `batch_generate.py`):

```text
outputs/wav|mp3/<run>/<character>/<id>.wav|mp3
outputs/logs/<run>_manifest.csv   üretim satırları     <run>_run.json   model + karakter parametreleri
outputs/logs/<run>_qc.csv         QC satırları          <run>_qc.json    QC özeti (karakter başına CER)
```

## Kurulum

Gereksinimler: Python 3.10, NVIDIA GPU (8 GB VRAM yeter), **ffmpeg** (`winget install Gyan.FFmpeg`), git.

```powershell
py -3.10 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
Copy-Item .env.example .env
git clone https://github.com/litagin02/Style-Bert-VITS2 third_party/Style-Bert-VITS2
```

> `pyopenjtalk-plus` ile SBV2'nin istediği `pyopenjtalk-dict` çakışır: SBV2 kurarken
> `pyopenjtalk-dict` gelirse kaldır, `pyopenjtalk` klasörünü sil, `pyopenjtalk-plus`'ı yeniden kur.

Model `models/finetuned/jvs_jvnv16/` altında olmalı (eğitim için aşağı bak). Varsayılan model
`configs/paths.yaml → models.default`.

## Kullanım

```bash
python scripts/generate_voice.py --list                                   # karakterler + model bilgisi
python scripts/generate_voice.py --text "こんにちは。" --character yuki --out hello.mp3
python scripts/run_pipeline.py --texts tests/test_sentences.json --characters all --run deneme
#   ^ N cümle × M karakter toplu üretim + ayrı süreçte Whisper QC + rapor (tek komut)
python scripts/qc_whisper.py --wav outputs/wav/hello.wav --text "こんにちは。"    # tek dosya QC
python scripts/check_reading.py "cümle"    # Open JTalk vs Sudachi okunuş anlaşmazlıkları -> lexicon.yaml'a ekle
python -m unittest discover -s tests       # birim testleri (model gerektirmez)
```

**Metin ön-yüzü:** her girdi okunur — normalize (tarih/saat/telefon/kesir/emoji),
`lexicon.yaml` sözlüğü (İngilizce marka vb.), `!` uyumluluk yaması, uzun metin cümle
sınırından parçalanır, 1–4 moralık tek kelimeler yavaşlatılır.

## Karakterler ve Duygu

`configs/characters.yaml`: `speaker_id`, `style_id`, `style_weight`, `length_scale` (hız),
`sdp_ratio` (tonlama çeşitliliği), `noise` (ifade), isteğe bağlı `model_dir`. Kontrol model
seviyesindedir; DSP pitch shift yok.

- **Kullanılan sesler (JVNV):** `yuki` (F1), `hana` (F2), `kenta` (M1), `takeshi` (M2) — duygulu diyalog
  verisiyle eğitildikleri için ifadeli. Taban stil `Neutral`.
- **JVS karakterleri** (`deep_narrator`, `bright_female`, … 10 preset) aynı modelde çalışır ama dinleme
  testinde belirgin kötü bulundu; sitede kullanılmıyor. Konuşmacı-embedding harmanlama denendi ("hep aynı"), silindi.
- **Stiller:** `Neutral, normal, whisper, falsetto` (JVS) + `happy, sad, surprise, anger, fear, disgust` (JVNV).
  Cümle bazlı stil `web/dialog.json`'daki `style` / `style_weight` alanlarından gelir
  (`synthesize_wav(..., style=, style_weight=)`); model o stili tanımıyorsa karakterin kendi stili kullanılır.

## Prozodi: soru tonlaması (`src/japonca_tts/prosody.py`)

Japonca bilgi soran soruda (〜ですか？) son mora yükselir. Elimizdeki hiçbir lisanslı veri bunu
içermiyor: JVS'te soru yok; JVNV'deki sorular retorik/şaşkınlık ve **insan kayıtlarında da düşüyor**
(ölçüldü: son mora −6 st). Model bu yüzden soruları bildirim gibi düşürüyordu; `given_tone`, JVNV
duygu modeli ve stil-vektör deltaları eğriyi değiştirmedi. Çözüm: yalnızca `？` ile biten parçanın
son 220 ms'inde WORLD vokoder ile F0 rampası (+5 st) ve 1.3× uzatma, ölçüp-doğrulayan döngüyle;
klibin geri kalanına dokunulmaz. Ayar: `configs/paths.yaml → prosody` (`question_rise: false` kapatır).

Ölçüm (8 karakter × 8 soru): soru sonu medyan −3.1 st (v1) → **+4.8 st** (v2), %97 klip ≥ +2.5 st.
Whisper QC (large-v3, 336 klip): 335/336, ort. CER 0.0065 (v1: 237/240, 0.010).

## Web Sitesi (`web/`) — iki mod

**Canlı sohbet (yerel):** `python scripts/serve_chat.py` → http://localhost:8000. Sayfa sunucuyu
görünce serbest yazma moduna geçer: ne yazarsan yaz (romaji, Türkçe, Japonca) Ollama'daki
`qwen3:8b` bunu doğal Japonca'ya çevirir, Türkçesini ve duygusunu çıkarır, kısa bir Japonca cevap
üretir; iki taraf da gerçek modelle anında seslendirilir (cümle bazlı duygu + soru tonlaması).
Gereksinim: [Ollama](https://ollama.com) çalışıyor ve `ollama pull qwen3:8b` (VRAM: TTS ~2 GB +
LLM 5.2 GB; 8 GB'a sığar, `--llm qwen3:4b` daha hafif). Tur süresi ~6 s (ilk tur +5 s model yükleme).
Japonca bilmek gerekmez: her balonda **romaji** satırı ve Türkçe çeviri var; her bot cevabından
sonra 3 **tıklanabilir cevap önerisi** (Japonca + romaji + Türkçe) gelir. Romaji `pykakasi` ile
(`chat.romaji`), öneriler LLM'den. Uç noktalar: `GET /api/health`, `POST /api/chat`, `POST /api/tts`.
Klipler `outputs/chat/`.

**Statik mod:** sunucu yoksa (örn. GitHub Pages) sayfa hazır cümle moduna düşer: bir cümle
seçilir, Enter ile gönderilir; önceden üretilmiş klipler çalınır. Model dışarı çıkmaz; yayınlamak
için `web/` klasörünü olduğu gibi kopyala. Canlı mod herkese açık bir sitede **çalışmaz** (model
ve LLM senin makinende).

```text
web/dialog.json    "characters": sitedeki sesler; "turns": cümle + cevap (ja, tr altyazı, style, style_weight)
web/index.html     sayfa (tek dosya; açık/koyu tema; ses seçici + önizleme; çalarken ses dalgası)
web/manifest.js    build_web.py üretir
web/audio/<karakter>/<tur>_{u,r}.mp3   önceden üretilmiş klipler (u = kullanıcı, r = cevap)
```

```bash
python scripts/build_web.py                 # eksik klipleri üret (var olanlar atlanır); --force hepsini
python -m http.server -d web 8000           # http://localhost:8000 (çift tıkla da çalışır; dalga formu yalnızca sunucudan gerçek)
python scripts/render_conversation.py --me yuki --them takeshi --out conversation.mp3   # 12 turluk tek parça kayıt
```


## Eğitim (yeniden üretmek / yeni konuşmacı eklemek)

Ayrı ortam: `py -3.10 -m venv .venv-train`, `pip install -r requirements-train.txt`,
`python scripts/download_train_models.py` (JP BERT + WavLM + JP-Extra base). Eğitim 8 GB VRAM'in
tamamını ister; `train_when_gpu_free.ps1` GPU boşalana kadar bekler. Başka süreç VRAM tutarsa
Windows paylaşımlı belleğe taşır ve hız ~100× düşer.

```powershell
# Veri: JVS (12 konuşmacı: 60 normal + tüm whisper/falsetto) + JVNV (NV kesilmiş, metin temizlenmiş)
& .\.venv\Scripts\python.exe scripts\download_jvs.py ; scripts\build_manifest.py ; scripts\preprocess_audio.py
& .\.venv\Scripts\python.exe scripts\prepare_sbv2_data.py -m jvs_jvnv16 --speakers jvs001 jvs003 jvs006 jvs028 jvs042 jvs071 jvs002 jvs017 jvs025 jvs036 jvs064 jvs084 --style-dirs --normal-per-speaker 60
& .\.venv\Scripts\python.exe scripts\prepare_jvnv_data.py -m jvs_jvnv16       # -> esd_jvnv.list; esd.list = jvs + jvnv (JVS satırları ÖNCE: spk id sırası korunur)

# SBV2 ön-işleme (repo kökünden, .venv-train ile)
Set-Location third_party\Style-Bert-VITS2 ; $P = "..\..\.venv-train\Scripts\python.exe"
& $P preprocess_all.py -m jvs_jvnv16 -b 2 -e 42 -s 500 --use_jp_extra --normalize --trim --num_processes 2 --yomi_error skip
& $P preprocess_text.py --config-path Data\jvs_jvnv16\config.json --transcription-path Data\jvs_jvnv16\esd.list --train-path Data\jvs_jvnv16\train.list --val-path Data\jvs_jvnv16\val.list --val-per-lang 6 --yomi_error skip --correct_path --use_jp_extra   # val boş kalmasın
Set-Location ..\..

# Devam eğitimi: konuşmacı sayısı artıyorsa checkpoint'i genişlet (yoksa SBV2 emb_g'yi rastgele sıfırlar)
& .\.venv-train\Scripts\python.exe scripts\expand_checkpoint.py --src <eski>\models --dst third_party\Style-Bert-VITS2\Data\jvs_jvnv16\models --n-speakers 16
powershell -ExecutionPolicy Bypass -File scripts\train_when_gpu_free.ps1 -Model jvs_jvnv16   # log: outputs/logs/train_<model>.log (UTF-16)
#   "Found the model. Current epoch is N" görülmeli. Çökerse aynı komut son checkpoint'ten devam eder.

# Kurulum + stil vektörleri
& .\.venv\Scripts\python.exe scripts\install_finetuned.py -m jvs_jvnv16
& .\.venv\Scripts\python.exe scripts\rebuild_style_vectors.py -m jvs_jvnv16   # ŞART: düz ortalama vektörler konuşmacı ofseti taşır (perde +3…+7 st kayar)
```

Son eğitim: 30→42 epoch, 2.506 klip (~317 dk), 1.36 it/s, ~3 sa 40 dk; son checkpoint
`third_party/Style-Bert-VITS2/Data/jvs_jvnv16/models/` (G/D/WD_31482) devam eğitimi için tutuluyor.

**Yeni ses eklemek için tek gerçek yol:** JVNV tarzı duygulu diyalog verisi olan yeni konuşmacılarla
devam eğitimi (ITA-korpusu okuyan ücretsiz ses aktörleri; lisans tek tek okunmalı). Düz okuma verisi
(JVS gibi) ifadeli ses vermiyor.

## Lisans / Hukuki

- **JVS** ticari OLMAYAN araştırma ve kişisel kullanım (blog dahil) içindir; ticari kullanım için resmi
  sayfadaki iletişim adreslerinden izin gerekir (https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus).
- **JVNV** CC BY-SA 4.0: atıf zorunlu; türev (model/ses) aynı lisansla paylaşılır.
- **Style-Bert-VITS2** kodu AGPL-3.0 (user_dict LGPL); kod yayınlanırsa AGPL uyumlu lisans seç.
- Eğitilen model özel kalır, ağırlıklar paylaşılmaz. Sitede yalnızca kod, açıklama ve üretilmiş MP3'ler
  (atıf: JVS + JVNV, Takamichi ve ark., Tokyo Üniversitesi; Style-Bert-VITS2, litagin02; "ticari olmayan, sentetik").
