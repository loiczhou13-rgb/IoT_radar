import numpy as np

class FMCWProcessor:
    def __init__(self, fs, bw, n_samples, chirp_tx, n_history):
        self.fs = fs
        self.bw = bw
        self.n_samples = n_samples
        self.chirp_tx = chirp_tx.astype(np.complex64)
        self.n_history = n_history
        self.background = None
        self.mti_alpha = 0.001

        # Résolution théorique FMCW : c / (2 * BW)
        self.dist_res = 299792458.0 / (2.0 * self.bw)

        # Fenêtres pour lisser les FFT affichées et le range profile
        self.win_range = np.blackman(self.n_samples).astype(np.float32)
        self.win_fft = np.hanning(self.n_samples).astype(np.float32)

        # Historique de variation, stocké directement en dB
        self.history = np.full((self.n_history, self.n_samples // 2), -200.0, dtype=np.float32)
        self.history_idx = 0
        self.history_filled = False

    @staticmethod
    def amp_to_db(x, floor_db=-200.0):
        """
        Convertit une amplitude complexe ou réelle en dB relatifs :
        max -> 0 dB, reste <= 0 dB.
        """
        mag = np.abs(x)
        mag = np.maximum(mag, 1e-12)
        db = 20.0 * np.log10(mag)
        db -= np.max(db)
        db = np.maximum(db, floor_db)
        return db

    def process_frame(self, rx_data):
        """
        Chaîne complète :
        1) synchro grossière
        2) dechirp
        3) FFT distance
        4) MTI
        5) variation
        6) FFT debug émis / reçu
        """
        rx_data = np.asarray(rx_data).astype(np.complex64).flatten()

        if rx_data.size < self.n_samples:
            return None, None, None, None, None, None

        # 1) Synchronisation grossière
        # On corrèle sur un buffer potentiellement plus long que n_samples.
        correlation = np.abs(np.correlate(rx_data, self.chirp_tx, mode="valid"))
        offset = int(np.argmax(correlation))

        if offset + self.n_samples > rx_data.size:
            return None, None, None, None, None, None

        rx_aligned = rx_data[offset: offset + self.n_samples]

        # 2) Déchirpage
        beat_complex = rx_aligned * np.conj(self.chirp_tx)

        # 3) Range FFT
        range_fft = np.fft.fft(beat_complex * self.win_range)[: self.n_samples // 2]

        # 4) MTI
        if self.background is None:
            self.background = range_fft.copy()
        else:
            self.background = (1.0 - self.mti_alpha) * self.background + self.mti_alpha * range_fft

        mti_complex = range_fft - self.background

        # Important : psd_mti est déjà en dB relatifs
        psd_mti = 20 * np.log10(np.abs(mti_complex) + 1e-12)

        # Axe distance
        dist_axis = np.arange(len(psd_mti), dtype=np.float32) * self.dist_res

        # 5) Historique + variation
        self.history[self.history_idx] = psd_mti.astype(np.float32)
        self.history_idx = (self.history_idx + 1) % self.n_history
        if self.history_idx == 0:
            self.history_filled = True

        if self.history_filled:
            variation = np.std(self.history, axis=0)
        else:
            used = self.history[:max(1, self.history_idx)]
            variation = np.std(used, axis=0) if used.shape[0] > 1 else np.zeros(len(psd_mti), dtype=np.float32)

        # 6) FFT de debug pour affichage
        fft_emis = np.fft.fftshift(np.fft.fft(self.chirp_tx * self.win_fft))
        fft_recu = np.fft.fftshift(np.fft.fft(rx_aligned * self.win_fft))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(self.n_samples, d=1.0 / self.fs))

        return dist_axis, psd_mti, variation, freq_axis, fft_emis, fft_recu

    def reset_background(self):
        self.background = None
        self.history.fill(-200.0)
        self.history_idx = 0
        self.history_filled = False