# PlutoDoppler/main.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from sdr_device import PlutoDevice
from signal_generator import SignalGenerator
from config import RadarConfig
from processor import RadarProcessor
from interface import RadarUI

class RadarApp:
    def __init__(self):
        self.hw = PlutoDevice(fs=RadarConfig.FS, lo=RadarConfig.FC)
        self.hw.sdr.rx_buffer_size = RadarConfig.BUFFER_SIZE
        self.hw.sdr.tx_hardwaregain_chan0 = -40 
        
        self.gain, self.sql, self.flip, self.paused = 55.0, -110, 1, False
        
        # Variables de curseur
        self.v_record = 0.0
        self.cursor_v = 0.0
        self.cursor_timer = 0 
        
        self.hw.set_rx_gain('manual', self.gain)
        self.processor = RadarProcessor()
        self.ui = RadarUI(self.on_key)
        
        self.hw.tx(SignalGenerator.generate('Sinus', 0, RadarConfig.FS, 10000))
        
        self.ani = FuncAnimation(self.ui.fig, self.update, interval=RadarConfig.UPDATE_MS, 
                                 blit=True, cache_frame_data=False)
        plt.show()

    def update(self, frame):
        if self.paused:
            return self.ui.img_hz, self.ui.img_ms, self.ui.txt, self.ui.txt_record, self.ui.gauge_bar, self.ui.sql_line, self.ui.peak_cursor

        try:
            rx = self.hw.rx()
            f_hz, v_ms, raw, mti = self.processor.process_frame(rx, self.flip)
            
            # --- LOGIQUE DE DÉTECTION STABILISÉE ---
            search_mask = (np.abs(v_ms) > 0.2) & (np.abs(v_ms) < RadarConfig.V_LIMIT)
            v_max_inst = 0.0
            
            if np.any(search_mask):
                idx = np.argmax(mti[search_mask])
                pwr_peak = mti[search_mask][idx]
                v_detected = v_ms[search_mask][idx]
                
                # RÈGLE 1 : Le pic doit dépasser le SQL pour être affiché
                if pwr_peak > self.sql:
                    v_max_inst = abs(v_detected)
                    if v_max_inst > self.v_record: self.v_record = v_max_inst
                
                # RÈGLE 2 : Le curseur ne se déplace que si le pic est "fort" (SQL + Marge)
                if pwr_peak > (self.sql + RadarConfig.DETECTION_MARGIN):
                    self.cursor_v = v_detected
                    self.cursor_timer = 50 # On garde le curseur 2.5 secondes
            
            # Gestion de l'opacité (Fade out si aucun nouveau pic fort)
            cursor_alpha = 0.0
            if self.cursor_timer > 0:
                # Le curseur reste opaque au début puis s'efface
                cursor_alpha = min(0.8, self.cursor_timer / 10.0)
                self.cursor_timer -= 1

            noise_floor = np.median(raw)
            return self.ui.update_displays(f_hz, v_ms, raw, mti, self.gain, self.sql, 
                                          v_max_inst, noise_floor, self.v_record, self.cursor_v, cursor_alpha)
        except Exception:
            return self.ui.img_hz, self.ui.img_ms, self.ui.txt, self.ui.txt_record, self.ui.gauge_bar, self.ui.sql_line, self.ui.peak_cursor

    def on_key(self, event):
        if event.key == 'pageup': self.gain = min(71, self.gain + 2)
        elif event.key == 'pagedown': self.gain = max(0, self.gain - 2)
        elif event.key == '+': self.sql += 2
        elif event.key == '-': self.sql -= 2
        elif event.key == 'd': self.flip *= -1
        elif event.key == 'r': 
            self.ui.reset_history()
            self.v_record = 0.0
            self.cursor_timer = 0
        elif event.key == ' ': self.paused = not self.paused
        elif event.key == 'q': plt.close()
        self.hw.set_rx_gain('manual', self.gain)

if __name__ == "__main__":
    RadarApp()