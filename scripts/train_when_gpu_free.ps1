# SBV2 fine-tune'u GPU boşalınca başlatır (MVP 3b).
#
# Neden: 8 GB VRAM'in tamamı eğitime lazım. Başka bir süreç (oyun, tarayıcı
# video, vs.) VRAM tutuyorsa Windows eğitimi "shared GPU memory"e (sistem RAM)
# taşırır ve hız ~100x düşer (ölçüldü: 2 s/adım -> 400 s/adım). Bu script
# nvidia-smi ile bekler, en az -MinFreeMiB boş olunca train_ms_jp_extra.py'yi
# çalıştırır. Eğitim otomatik resume eder (Data\<model>\models\G_*.pth).
#
# Kullanım (repo kökünden):
#   powershell -ExecutionPolicy Bypass -File scripts\train_when_gpu_free.ps1 -Model jvs_multi12
#   powershell ... -Model jvs_multi12 -MinFreeMiB 7000 -PollSec 30
param(
    [string]$Model = "jvs_multi12",
    [int]$MinFreeMiB = 7000,
    [int]$PollSec = 30
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Repo = Join-Path $Root "third_party\Style-Bert-VITS2"
$Py   = Join-Path $Root ".venv-train\Scripts\python.exe"
$Log  = Join-Path $Root "outputs\logs\train_$Model.log"

function Get-FreeMiB {
    $v = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    return [int]($v.Trim())
}

$waited = 0
while ((Get-FreeMiB) -lt $MinFreeMiB) {
    if ($waited % 300 -eq 0) {
        $free = Get-FreeMiB
        Write-Host ("[{0}] GPU bos degil ({1} MiB bos, {2} gerekli) - bekleniyor..." -f (Get-Date -Format "HH:mm:ss"), $free, $MinFreeMiB)
    }
    Start-Sleep -Seconds $PollSec
    $waited += $PollSec
}

Write-Host ("[{0}] GPU bos ({1} MiB). Egitim basliyor: {2}" -f (Get-Date -Format "HH:mm:ss"), (Get-FreeMiB), $Model)
Set-Location $Repo
& $Py train_ms_jp_extra.py --config "Data\$Model\config.json" --model "Data\$Model" --assets_root model_assets 2>&1 |
    Tee-Object -FilePath $Log -Append
Write-Host ("[{0}] Egitim sureci bitti (exit {1})." -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE)
exit $LASTEXITCODE
