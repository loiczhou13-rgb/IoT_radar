# PlutoDoppler/processor.py
import numpy as np
from config import RadarConfig

class RadarProcessor:
    def __init__(self):
        self.prev_rx = None
        self.psd_avg = None

    def process_frame(self, rx, direction_flip=1):
        # 1. Nettoyage DC et MTI
        rx_clean = rx - np.mean(rx)
        if self.prev_rx is not None:
            rx_proc = rx_clean - self.prev_rx
        else:
            rx_proc = rx_clean
        self.prev_rx = rx_clean.copy()

        # 2. FFT avec Zero-Padding pour la HD
        # On applique la fenêtre AVANT d'ajouter les zéros
        window = np.blackman(len(rx_proc))
        rx_windowed = rx_proc * window
        
        # FFT sur 131072 points (Padding automatique par numpy)
        f_hz = np.fft.fftshift(np.fft.fftfreq(RadarConfig.FFT_RESOLVED, 1/RadarConfig.FS))
        v_ms = direction_flip * (f_hz * RadarConfig.LAMBDA) / 2
        
        # Calcul des spectres
        fft_raw = np.fft.fft(rx_clean * np.blackman(len(rx_clean)), n=RadarConfig.FFT_RESOLVED)
        psd_raw = np.fft.fftshift(20*np.log10(np.abs(fft_raw)/len(rx_clean) + 1e-12))
        
        fft_mti = np.fft.fft(rx_windowed, n=RadarConfig.FFT_RESOLVED)
        psd_mti = np.fft.fftshift(20*np.log10(np.abs(fft_mti)/len(rx_clean) + 1e-12))
        
        if self.psd_avg is None: self.psd_avg = psd_mti
        else: self.psd_avg = self.psd_avg * 0.3 + psd_mti * 0.7 # Plus réactif
        
        return f_hz, v_ms, psd_raw, self.psd_avg