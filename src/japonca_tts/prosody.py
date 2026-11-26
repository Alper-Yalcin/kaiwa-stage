"""
Soru cümlesi son-mora yükselmesi (MVP 6).

Neden: Japonca bilgi soran soruda (〜ですか？) son mora yükselir. Elimizdeki
eğitim verilerinin hiçbiri bunu içermiyor: JVS'te soru yok; JVNV'deki sorular
duygusal/retorik ve insan kayıtlarında da düşüyor (ölçüldü: son mora −6 st).
Model bu yüzden her soruyu bildirim gibi düşürüyor. Bu modül yalnızca "？" ile
biten parçanın SON SESLİ BÖLGESİNDE (son mora) F0'ı rampayla yükseltir ve hafif
uzatır; WORLD vokoder spektral zarfı koruduğu için ses rengi değişmez. Klibin
geri kalanına dokunulmaz (kuyruk kesilir, işlenir, çapraz geçişle geri eklenir).

Karakter kontrolü için DSP kullanılmaz (yol haritası); bu, tek bir dilbilgisel
kuralın hedefli düzeltmesidir ve varsayılan olarak yalnızca soru parçalarına uygulanır.
"""
from __future__ import annotations

import numpy as np

FRAME_MS = 5.0
TAIL_SEC = 0.22            # son mora olarak ele alınan kuyruk (yükselme rampası burada)
PLAUSIBLE_ST = 10.0        # medyandan bu kadar uzak F0 = izleme hatası / ünsüz -> sessiz sayılır
XFADE_SEC = 0.02


def _plausible(f0: np.ndarray) -> np.ndarray:
    """Sesli ve izleme-hatasız çerçeve maskesi (44.1 kHz TTS çıkışında harvest
    ünsüzleri de 'sesli' işaretleyip oktav hatası verebiliyor)."""
    v = f0 > 0
    if not v.any():
        return v
    med = np.median(f0[v])
    with np.errstate(divide="ignore"):
        dev = np.abs(12.0 * np.log2(np.where(v, f0, med) / med))
    return v & (dev < PLAUSIBLE_ST)


def measure_rise(audio: np.ndarray, sr: int) -> float:
    """Son 60 ms'nin, kuyruktan önceki 400 ms'ye göre yarım ton farkı (algoritmanın hedefiyle aynı tanım)."""
    import pyworld as pw
    x = audio.astype(np.float64)
    f0, t = pw.harvest(x, sr, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    ok = _plausible(f0)
    if ok.sum() < 20:
        return float("nan")
    f0 = np.where(ok, f0, 0.0)
    e = int(np.max(np.where(ok)[0])) + 1
    s = e - int(TAIL_SEC * 1000 / FRAME_MS)
    end = f0[e - 12:e]; end = end[end > 0]
    prev = f0[max(0, s - 80):s]; prev = prev[prev > 0]
    if len(end) < 3 or len(prev) < 5:
        return float("nan")
    return float(12.0 * np.log2(np.median(end) / np.median(prev)))


def apply_question_rise(audio: np.ndarray, sr: int, target_st: float = 4.0,
                        stretch: float = 1.25) -> np.ndarray:
    """Son moranın sonunu, öncesindeki 400 ms medyanına göre +target_st'ye (ve en az
    cümle medyanının +3 st üstüne) taşır. Sonucu ölçer; hedefin 1.5 st altındaysa
    farkı ekleyip en fazla 2 kez yeniden dener.

    audio: float32 mono. Uygun sesli bölge bulunamazsa ses değişmeden döner.
    """
    extra = 0.0
    best, best_r = audio, measure_rise(audio, sr)
    for _ in range(3):
        out = _rise_once(audio, sr, target_st + extra, stretch)
        r = measure_rise(out, sr)
        if not np.isnan(r) and (np.isnan(best_r) or r > best_r):
            best, best_r = out, r
        if np.isnan(r) or r >= target_st - 1.5:
            break
        extra += (target_st - r)
    return best


def _rise_once(audio: np.ndarray, sr: int, target_st: float, stretch: float) -> np.ndarray:
    import pyworld as pw

    x = audio.astype(np.float64)
    f0, t = pw.harvest(x, sr, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    fpb = 1000.0 / FRAME_MS
    ok = _plausible(f0)
    if ok.sum() < int(0.15 * fpb):
        return audio
    f0 = np.where(ok, f0, 0.0)                        # kuyruk sentezinde çöp F0 kullanılmasın
    med = np.median(f0[ok])
    e = int(np.max(np.where(ok)[0])) + 1              # konuşmanın (sesli) sonu
    s = max(0, e - int(TAIL_SEC * fpb))
    if ok[s:e].sum() < int(0.06 * fpb):
        return audio
    prev = f0[max(0, s - int(0.4 * fpb)):s]
    prev = prev[prev > 0]
    ref = np.median(prev) if len(prev) >= 5 else med
    end_now = np.median(f0[e - 8:e][f0[e - 8:e] > 0]) if (f0[e - 8:e] > 0).any() else np.median(f0[s:e][f0[s:e] > 0])
    target_hz = max(ref * 2 ** (target_st / 12.0), med * 2 ** (3.0 / 12.0))
    delta_st = 12.0 * np.log2(target_hz / end_now)
    if delta_st <= 0.5:                               # zaten yükseliyor
        return audio

    # Kuyruk: [s - margin, sonuna kadar]; margin önceki sessiz/ünsüz bölgeyi de alsın ki
    # WORLD çerçevesi tutarlı olsun. Rampayı yalnızca [s, e) içinde uygula.
    margin = int(0.08 * fpb)
    a = max(0, s - margin)
    f0_t, t_t = f0[a:].copy(), t[a:] - t[a]
    x_t = x[int(a * FRAME_MS * sr / 1000):]
    sp = pw.cheaptrick(x_t, f0_t, t_t, sr)
    ap = pw.d4c(x_t, f0_t, t_t, sr)
    ls, le = s - a, e - a
    ramp = np.zeros_like(f0_t)
    n = max(le - ls, 1)
    ramp[ls:le] = np.linspace(0.0, 1.0, n) ** 1.3       # sona doğru hızlanan yükselme
    ramp[le:] = 1.0
    f0_new = f0_t * (2.0 ** (delta_st * ramp / 12.0))

    # Hafif uzatma: yalnızca son mora çerçeveleri tekrar edilir (interpolasyon).
    voiced = (f0_t > 0).astype(float)
    if stretch > 1.0:
        src_idx = np.arange(len(f0_t))
        idx = np.concatenate([np.arange(0, ls), np.linspace(ls, len(f0_t) - 1, int((len(f0_t) - ls) * stretch))])
        f0_new = np.interp(idx, src_idx, f0_new)
        voiced = np.interp(idx, src_idx, voiced)
        sp = np.stack([np.interp(idx, src_idx, sp[:, k]) for k in range(sp.shape[1])], axis=1)
        ap = np.stack([np.interp(idx, src_idx, ap[:, k]) for k in range(ap.shape[1])], axis=1)
    f0_new = np.where(voiced > 0.5, f0_new, 0.0)
    y_t = pw.synthesize(np.ascontiguousarray(f0_new), np.ascontiguousarray(sp),
                        np.ascontiguousarray(ap), sr, frame_period=FRAME_MS).astype(np.float32)

    # Baş kısım orijinal, kuyruk yeni; çapraz geçiş.
    head_len = int(a * FRAME_MS * sr / 1000)
    nx = int(XFADE_SEC * sr)
    head = audio[:head_len].astype(np.float32)
    if len(head) >= nx and len(y_t) >= nx:
        w = np.linspace(0, 1, nx, dtype=np.float32)
        joined = np.concatenate([head[:-nx], head[-nx:] * (1 - w) + y_t[:nx] * w, y_t[nx:]])
    else:
        joined = np.concatenate([head, y_t])
    # WORLD çıkışı seviye farkı yapabilir; kuyruğu orijinal RMS'e eşle.
    ref_rms = np.sqrt(np.mean(audio[head_len:] ** 2) + 1e-9)
    new_rms = np.sqrt(np.mean(y_t ** 2) + 1e-9)
    joined[head_len:] *= float(np.clip(ref_rms / new_rms, 0.5, 2.0))
    return np.clip(joined, -1.0, 1.0)
