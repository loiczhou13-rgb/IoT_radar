import adi

class PlutoDevice:
    def __init__(self, ip="ip:192.168.2.1", fs=2_000_000, lo=800_000_000):
        try:
            self.sdr = adi.Pluto(ip)
            self.fs = int(fs)
            self.sdr.sample_rate = self.fs
            self.set_lo(lo)
            self.sdr.rx_buffer_size = 32768
            self.sdr.rx_gain_control_mode_chan0 = 'slow_attack'
            self.sdr.tx_gain_control_mode_chan0 = 'slow_attack'
            self.rx_gain = 30
            self.tx_gain = -20
        except Exception as e:
            print(f"ERREUR MATÉRIELLE : {e}"); exit()

    def set_lo(self, lo):
        self.sdr.rx_lo = int(lo)
        self.sdr.tx_lo = int(lo)

    def set_rx_gain(self, mode, val=None):
        self.sdr.rx_gain_control_mode_chan0 = mode
        if mode == 'manual' and val is not None:
            self.rx_gain = val
            self.sdr.rx_hardwaregain_chan0 = int(self.rx_gain)

    def set_tx_gain(self, mode, val=None):
        self.sdr.tx_gain_control_mode_chan0 = mode
        if mode == 'manual' and val is not None:
            self.tx_gain = val
            self.sdr.tx_hardwaregain_chan0 = int(self.tx_gain)

    def rx(self):
        return self.sdr.rx() / (2**11)

    def tx(self, iq):
        self.sdr.tx_destroy_buffer() # On nettoie l'ancien signal
        self.sdr.tx_cyclic_buffer = True # Indispensable pour le câble !
        self.sdr.tx(iq * (2**11))

    def stop_tx(self):
        self.sdr.tx_destroy_buffer()