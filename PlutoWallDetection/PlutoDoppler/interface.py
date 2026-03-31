# PlutoDoppler/interface.py
import numpy as np
import matplotlib.pyplot as plt
from config import RadarConfig

class RadarUI:
    def __init__(self, key_callback):
        plt.style.use('dark_background')
        self.fig, (self.ax_hz, self.ax_ms) = plt.subplots(2, 1, figsize=(15, 10))
        self.fig.canvas.manager.set_window_title("RADAR DOPPLER MASTER - V7.0 (FINAL)")

        self.data_hz = np.full((RadarConfig.N_HISTORY, RadarConfig.WATERFALL_RES), -115.0)
        self.data_ms = np.full((RadarConfig.N_HISTORY, RadarConfig.WATERFALL_RES), -115.0)
        
        # Calcul de la durée totale de l'historique en secondes
        self.time_span = (RadarConfig.N_HISTORY * RadarConfig.UPDATE_MS) / 1000.0

        # --- 1. WATERFALL Hz (Temps réel sur l'axe Y) ---
        self.img_hz = self.ax_hz.imshow(self.data_hz, aspect='auto', animated=True,
                                       extent=[-RadarConfig.HZ_LIMIT, RadarConfig.HZ_LIMIT, -self.time_span, 0],
                                       cmap='viridis', vmin=RadarConfig.DB_MIN, vmax=RadarConfig.DB_MAX)
        self.ax_hz.set_title("ANALYSE TECHNIQUE (Hz)", color='cyan', fontsize=12, loc='left')
        self.ax_hz.set_ylabel("Temps (sec)", fontsize=10)

        # --- 2. WATERFALL Vitesse (Temps réel sur l'axe Y) ---
        self.img_ms = self.ax_ms.imshow(self.data_ms, aspect='auto', animated=True,
                                       extent=[-RadarConfig.V_LIMIT, RadarConfig.V_LIMIT, -self.time_span, 0],
                                       cmap='magma', vmin=RadarConfig.DB_MIN, vmax=RadarConfig.DB_MAX)
        self.ax_ms.set_title("MESURE DE VITESSE (m/s)", color='orange', fontsize=12, loc='left')
        self.ax_ms.set_ylabel("Temps (sec)", fontsize=10)
        
        # --- 3. CURSEUR DE CRÊTE (Persistent) ---
        self.peak_cursor = self.ax_ms.axvline(0, color='yellow', alpha=0, lw=1.5, ls='--')
        
        # --- 4. TEXTES DE STATUT ET RECORD ---
        self.txt = self.ax_hz.text(0.01, 0.93, "", transform=self.ax_hz.transAxes, 
                                  color='#00ff41', family='monospace', fontsize=10, fontweight='bold')
        
        self.txt_record = self.ax_ms.text(0.98, 0.05, "RECORD: 0.00 m/s", transform=self.ax_ms.transAxes,
                                         color='red', family='monospace', fontsize=11, fontweight='bold',
                                         ha='right', bbox=dict(facecolor='black', alpha=0.6))

        # Jauge et Commandes (Identique V6.1)
        self.ax_gauge = self.fig.add_axes([0.88, 0.15, 0.02, 0.4]) 
        self.gauge_bar = self.ax_gauge.bar(0, 0, color='#00d4ff', alpha=0.6)[0]
        self.sql_line = self.ax_gauge.axhline(RadarConfig.DB_MIN, color='red', lw=2)
        self.ax_gauge.set_ylim(RadarConfig.DB_MIN, RadarConfig.DB_MAX)
        self.ax_gauge.set_xticks([])

        self.fig.subplots_adjust(left=0.07, right=0.85, top=0.92, bottom=0.08, hspace=0.3)
        self.fig.canvas.mpl_connect('key_press_event', key_callback)

    def update_displays(self, f_hz, v_ms, psd_raw, psd_mti, gain, sql, v_max, noise_floor, v_record, cursor_v, cursor_alpha):
        # Update Images
        mask_hz = (f_hz >= -RadarConfig.HZ_LIMIT) & (f_hz <= RadarConfig.HZ_LIMIT)
        res_hz = np.interp(np.linspace(-RadarConfig.HZ_LIMIT, RadarConfig.HZ_LIMIT, RadarConfig.WATERFALL_RES), 
                          f_hz[mask_hz], psd_raw[mask_hz])
        self.data_hz = np.roll(self.data_hz, -1, axis=0); self.data_hz[-1, :] = res_hz
        self.img_hz.set_array(self.data_hz)

        mask_v = (v_ms >= -RadarConfig.V_LIMIT) & (v_ms <= RadarConfig.V_LIMIT)
        res_v = np.interp(np.linspace(-RadarConfig.V_LIMIT, RadarConfig.V_LIMIT, RadarConfig.WATERFALL_RES), 
                         v_ms[mask_v], psd_mti[mask_v])
        res_v[RadarConfig.WATERFALL_RES//2-3 : RadarConfig.WATERFALL_RES//2+3] = -115.0 # Zone morte réduite
        res_v[res_v < sql] = -115.0
        self.data_ms = np.roll(self.data_ms, -1, axis=0); self.data_ms[-1, :] = res_v
        self.img_ms.set_array(self.data_ms)

        # Update Curseur de crête
        self.peak_cursor.set_xdata([cursor_v, cursor_v])
        self.peak_cursor.set_alpha(cursor_alpha)

        # Update Jauge et Textes
        self.gauge_bar.set_height(noise_floor - RadarConfig.DB_MIN)
        self.gauge_bar.set_y(RadarConfig.DB_MIN)
        self.sql_line.set_ydata([sql, sql])
        
        self.txt.set_text(f"GAIN: {gain:.1f} dB | SQL: {sql} dB | V_PEAK: {v_max:.2f} m/s")
        self.txt_record.set_text(f"MAX SESSION: {v_record:.2f} m/s")
        
        return self.img_hz, self.img_ms, self.txt, self.txt_record, self.gauge_bar, self.sql_line, self.peak_cursor

    def reset_history(self):
        self.data_hz.fill(-115.0); self.data_ms.fill(-115.0)