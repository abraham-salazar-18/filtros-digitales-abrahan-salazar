# -*- coding: utf-8 -*-
# ============================================================
# ACTIVIDAD SUMATIVA N.º 4 — EL FILTRO PERFECTO
# Diseño y evaluación comparativa de filtros FIR y adaptativos LMS
# para la eliminación de ruido en señales de audio de voz.
#
# Script COMPLETO y REPRODUCIBLE (semillas fijas):
#   1) Prepara el audio (48 kHz -> 16 kHz, mono, normalizado)
#   2) Prueba 1: ruido blanco gaussiano + filtro FIR (Hamming)
#   3) Prueba 2: cancelación adaptativa LMS y comparación FIR vs LMS
#   4) Genera todos los audios, gráficas y un ZIP con el portafolio
# Ejecutar en UNA sola celda de Google Colab.
# ============================================================
import os, shutil, time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
try:
    import librosa
except ImportError:
    os.system("pip install -q librosa")
    import librosa

# ------------------------- PARÁMETROS -------------------------
FS = 16000            # frecuencia de trabajo (Hz)
SNR_DB = 5            # SNR inicial deseado (dB)
ORDEN_FIR = 100       # orden del FIR (101 coeficientes)
FC = 3400             # frecuencia de corte del FIR (Hz)
L_LMS = 16            # longitud del filtro LMS
C_MU = 0.05           # mu = C_MU / (L * potencia_referencia)
MEDIR_TIEMPOS = False # True = repite la medición de tiempos (Paso 6)
RUTA_AUDIO = None     # None = pedirá subir el archivo en Colab

SEN, EVI = "portafolio/senales/", "portafolio/evidencias/"
os.makedirs(SEN, exist_ok=True); os.makedirs(EVI, exist_ok=True)

def guardar(nombre):
    plt.tight_layout(); plt.savefig(EVI + nombre, dpi=200); plt.show()

def metricas(ref, ent, sal):
    """SNR y MSE respecto a la señal original (mismas muestras)."""
    e_in, e_out = ent - ref, sal - ref
    snr_in = 10 * np.log10(np.mean(ref**2) / np.mean(e_in**2))
    snr_out = 10 * np.log10(np.mean(ref**2) / np.mean(e_out**2))
    return snr_in, snr_out, snr_out - snr_in, np.mean(e_in**2), np.mean(e_out**2)

def lms(ref, d, L, mu):
    """Cancelador adaptativo LMS: salida = e[n] = d[n] - y[n]."""
    w = np.zeros(L)
    out = np.zeros(len(d))
    for n in range(L - 1, len(d)):
        x = ref[n - L + 1:n + 1][::-1]
        y = np.dot(w, x)
        e = d[n] - y
        w += mu * e * x
        out[n] = e
    return out

def psd_db(x):
    f, p = signal.welch(x, fs=FS, nperseg=2048)
    return f, 10 * np.log10(np.maximum(p, 1e-15))

# ======================== 1. AUDIO ========================
if RUTA_AUDIO is None:
    from google.colab import files
    subido = files.upload()
    RUTA_AUDIO = next(iter(subido))
audio, fs_orig = librosa.load(RUTA_AUDIO, sr=None, mono=True)
audio = librosa.resample(audio, orig_sr=fs_orig, target_sr=FS)
audio = audio / np.max(np.abs(audio))
N = len(audio)
sf.write(SEN + "audio_original.wav", audio, FS, subtype="FLOAT")
print("========== AUDIO ==========")
print("Archivo:", RUTA_AUDIO, "| fs original:", fs_orig, "Hz | fs trabajo:", FS, "Hz")
print("Duración:", round(N / FS, 4), "s | Muestras:", N, "(original: 612376)")

t = np.arange(N) / FS
plt.figure(figsize=(14, 4)); plt.plot(t, audio)
plt.title("Señal de audio original"); plt.xlabel("Tiempo (segundos)"); plt.ylabel("Amplitud"); plt.grid(True)
guardar("01_senal_original.png")

# ================= 2. PRUEBA 1: RUIDO BLANCO + FIR =================
np.random.seed(42)
pot_ruido = np.mean(audio**2) / (10 ** (SNR_DB / 10))
ruido_A = np.random.normal(0, np.sqrt(pot_ruido), N)
cont_A = audio + ruido_A
sf.write(SEN + "audio_contaminado.wav", cont_A, FS, subtype="FLOAT")
print("\n========== RUIDO BLANCO GAUSSIANO (prueba 1) ==========")
print("SNR real de entrada:", round(10*np.log10(np.mean(audio**2)/np.mean(ruido_A**2)), 2), "dB")
print("MSE inicial:", round(np.mean((audio - cont_A)**2), 6))

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax[0].plot(t, audio); ax[0].set_title("Señal de voz original")
ax[1].plot(t, cont_A); ax[1].set_title("Señal de voz contaminada con ruido blanco gaussiano")
for a in ax: a.set_ylabel("Amplitud"); a.grid(True)
ax[1].set_xlabel("Tiempo (segundos)")
guardar("02_original_vs_contaminada.png")

h = signal.firwin(numtaps=ORDEN_FIR + 1, cutoff=FC, window="hamming", pass_zero="lowpass", fs=FS)
ret = ORDEN_FIR // 2
fir_A = signal.lfilter(h, [1.0], cont_A)
ref_A, ent_A, sal_A = audio[:-ret], cont_A[:-ret], fir_A[ret:]
mA = metricas(ref_A, ent_A, sal_A)
sf.write(SEN + "audio_filtrado_FIR.wav", sal_A, FS, subtype="FLOAT")
print("\n========== FILTRO FIR (prueba 1) ==========")
print(f"Orden {ORDEN_FIR} | {ORDEN_FIR+1} coef. | Hamming | fc {FC} Hz | retardo {ret} muestras")
print(f"SNR entrada: {mA[0]:.2f} dB | SNR salida: {mA[1]:.2f} dB | Mejora: {mA[2]:.2f} dB")
print(f"MSE entrada: {mA[3]:.6f} | MSE salida: {mA[4]:.6f}")

w, H = signal.freqz(h, worN=4096, fs=FS)
plt.figure(figsize=(12, 4.5)); plt.plot(w, 20*np.log10(np.maximum(np.abs(H), 1e-10)))
plt.axvline(FC, color="gray", linestyle="--", label="Frecuencia de corte (3.400 Hz)")
plt.title("Respuesta en frecuencia del filtro FIR Hamming (orden 100)")
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Magnitud (dB)")
plt.xlim(0, 8000); plt.ylim(-120, 5); plt.grid(True); plt.legend()
guardar("03_FIR_respuesta_frecuencia.png")

tA = np.arange(len(sal_A)) / FS
fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for a, x, ti in zip(ax, [ref_A, ent_A, sal_A],
        ["Señal de voz original", "Señal de voz contaminada", "Señal procesada mediante filtro FIR"]):
    a.plot(tA, x); a.set_title(ti); a.set_ylabel("Amplitud"); a.grid(True)
ax[-1].set_xlabel("Tiempo (segundos)")
guardar("04_FIR_temporal.png")

plt.figure(figsize=(12, 6))
for x, lab in [(ref_A, "Voz original"), (ent_A, "Voz contaminada (ruido blanco)"), (sal_A, "Salida FIR")]:
    f_, p_ = psd_db(x); plt.plot(f_, p_, label=lab, linewidth=1.3)
plt.axvline(FC, color="gray", linestyle="--", label="Corte FIR (3400 Hz)")
plt.title("Densidad espectral de potencia antes y después del filtro FIR (prueba 1)")
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("PSD (dB/Hz)"); plt.legend(); plt.grid(True)
guardar("05_FIR_espectro_antes_despues.png")

# ============ 3. PRUEBA 2: LMS Y COMPARACIÓN EN IGUALDAD ============
rng = np.random.default_rng(123)
fuente = rng.normal(0, 1, N)
canal = [1.0, 0.5, 0.25]
ruido_sin = signal.lfilter(canal, [1.0], fuente)
factor = np.sqrt((np.mean(audio**2) / (10 ** (SNR_DB / 10))) / np.mean(ruido_sin**2))
ruido_B = ruido_sin * factor
ref_ruido = fuente * factor
cont_B = audio + ruido_B
pot_ref = np.mean(ref_ruido**2)
ini = FS

print("\n========== AJUSTE DEL PASO DE ADAPTACIÓN LMS ==========")
salidas = {}
for c in (0.5, C_MU):
    mu = c / (L_LMS * pot_ref)
    out = lms(ref_ruido, cont_B, L_LMS, mu)
    salidas[c] = out
    m = metricas(audio[ini:], cont_B[ini:], out[ini:])
    print(f"c = {c:<5} mu = {mu:.4f} | SNR in {m[0]:.2f} | SNR out {m[1]:.2f} | Mejora {m[2]:.2f} dB | "
          f"MSE in {m[3]:.6f} | MSE out {m[4]:.6f}")
mu_final = C_MU / (L_LMS * pot_ref)
lms_out = salidas[C_MU]
sf.write(SEN + "audio_filtrado_LMS.wav", lms_out, FS, subtype="FLOAT")
sf.write(SEN + "audio_contaminado_LMS.wav", cont_B, FS, subtype="FLOAT")

fir_B = signal.lfilter(h, [1.0], cont_B)
fin = N - ret
v_ref, v_ent = audio[ini:fin], cont_B[ini:fin]
s_fir, s_lms = fir_B[ini + ret:fin + ret], lms_out[ini:fin]
mF, mL = metricas(v_ref, v_ent, s_fir), metricas(v_ref, v_ent, s_lms)
sf.write(SEN + "comparacion_FIR.wav", s_fir, FS, subtype="FLOAT")
sf.write(SEN + "comparacion_LMS.wav", s_lms, FS, subtype="FLOAT")

print("\n==============================================")
print(" COMPARACIÓN EXPERIMENTAL FIR VS. LMS (misma señal)")
print("==============================================")
print(f"mu LMS = {mu_final} | evaluación: desde t = 1 s hasta N-{ret}")
for nombre, m in (("FIR", mF), ("LMS", mL)):
    print(f"--- {nombre} --- SNR in {m[0]:.2f} | SNR out {m[1]:.2f} | Mejora {m[2]:.2f} dB | "
          f"MSE in {m[3]:.6f} | MSE out {m[4]:.6f}")
print("Mayor mejora de SNR:", "LMS" if mL[2] > mF[2] else "FIR", "| Menor MSE:", "LMS" if mL[4] < mF[4] else "FIR")

# ---- verificación contra la corrida original ----
esperado = {"FIR prueba 1 (SNR out)": (mA[1], 8.61), "FIR comparación (SNR out)": (mF[1], 6.28),
            "LMS comparación (SNR out)": (mL[1], 16.23)}
print("\n========== ¿COINCIDE CON LA CORRIDA ORIGINAL? ==========")
for k, (obt, esp) in esperado.items():
    print(f"{k}: obtenido {obt:.2f} | original {esp} ->", "COINCIDE" if abs(obt - esp) <= 0.02 else "DIFIERE")

# ---- gráficas de la prueba 2 ----
tL = np.arange(N) / FS
fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for a, x, ti in zip(ax, [audio, cont_B, lms_out],
        ["Señal de voz original", "Señal de voz contaminada", "Señal recuperada mediante el filtro LMS"]):
    a.plot(tL, x); a.set_title(ti); a.set_ylabel("Amplitud"); a.grid(True)
ax[-1].set_xlabel("Tiempo (s)")
guardar("06_LMS_temporal.png")

tc = ini / FS + np.arange(len(v_ref)) / FS     # eje de tiempo real (desde t = 1 s)
fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for a, x, ti in zip(ax, [v_ref, v_ent, s_fir, s_lms],
        ["Señal de voz original", "Señal de voz contaminada",
         "Señal recuperada mediante FIR", "Señal recuperada mediante LMS"]):
    a.plot(tc, x); a.set_title(ti); a.set_ylabel("Amplitud"); a.grid(True)
ax[-1].set_xlabel("Tiempo (s) — tramo evaluado, desde t = 1 s")
guardar("07_comparacion_temporal_FIR_LMS.png")

plt.figure(figsize=(12, 6))
for x, lab, lw in [(v_ref, "Voz original", 1.5), (v_ent, "Voz contaminada", 1.2),
                   (s_fir, "Salida FIR", 1.2), (s_lms, "Salida LMS", 1.2)]:
    f_, p_ = psd_db(x); plt.plot(f_, p_, label=lab, linewidth=lw)
plt.axvline(FC, color="gray", linestyle="--", label="Corte FIR (3400 Hz)")
plt.title("Densidad espectral de potencia (Welch)")
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("PSD (dB/Hz)"); plt.legend(); plt.grid(True)
guardar("08_comparacion_espectro_PSD.png")

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
for a, (x, ti) in zip(ax.ravel(), [(v_ref, "Voz original"), (v_ent, "Voz contaminada"),
                                   (s_fir, "Salida FIR"), (s_lms, "Salida LMS")]):
    a.specgram(x, NFFT=512, Fs=FS, noverlap=256, cmap="viridis", vmin=-120, vmax=-20,
               xextent=(ini / FS, ini / FS + len(x) / FS))
    a.set_title(ti); a.set_xlabel("Tiempo (s)"); a.set_ylabel("Frecuencia (Hz)")
guardar("09_comparacion_espectrogramas.png")

# ================= 4. COMPLEJIDAD (opcional) =================
if MEDIR_TIEMPOS:
    t_fir, t_lms = [], []
    for _ in range(5):
        t0 = time.perf_counter(); signal.lfilter(h, [1.0], cont_B); t_fir.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); lms(ref_ruido, cont_B, L_LMS, mu_final); t_lms.append(time.perf_counter() - t0)
    print("\n========== COMPLEJIDAD COMPUTACIONAL ==========")
    print(f"FIR: {np.mean(t_fir):.4f} s (desv. {np.std(t_fir):.4f}) | {ORDEN_FIR+1} mult./muestra")
    print(f"LMS: {np.mean(t_lms):.4f} s (desv. {np.std(t_lms):.4f}) | ~{2*L_LMS+1} mult./muestra")
    print(f"Relación LMS/FIR: {np.mean(t_lms)/np.mean(t_fir):.1f}x")

# ======================= 5. EMPAQUETAR =======================
shutil.make_archive("portafolio_evidencias", "zip", "portafolio")
print("\nArchivos generados:")
for carpeta in (SEN, EVI):
    for f in sorted(os.listdir(carpeta)): print("  ", carpeta + f)
try:
    from google.colab import files
    files.download("portafolio_evidencias.zip")
except ImportError:
    pass
