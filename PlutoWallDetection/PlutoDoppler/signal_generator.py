# PlutoDoppler/signal_generator.py
import numpy as np

class SignalGenerator:

    @staticmethod
    def sinus(freq, fs, num_samples):
        t = np.arange(num_samples) / fs
        return np.exp(1j * 2 * np.pi * freq * t)

    @staticmethod
    def carre(freq, fs, num_samples):
        t = np.arange(num_samples) / fs
        return np.sign(np.cos(2*np.pi*freq*t)) + 1j*np.sign(np.sin(2*np.pi*freq*t))

    @staticmethod
    def triangle(freq, fs, num_samples):
        t = np.arange(num_samples) / fs
        return (2*np.abs(2*(t*freq-np.floor(t*freq+0.5)))-1) + 1j*(2*np.abs(2*(t*freq-np.floor(t*freq+0.25)))-1)

    @staticmethod
    def scie(freq, fs, num_samples):
        t = np.arange(num_samples) / fs
        return 2*(t*freq - np.floor(0.5 + t*freq)) + 1j*2*(t*freq - np.floor(0.25 + t*freq))

    @staticmethod
    def bruit(freq, fs, num_samples):
        return (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) / 3

    @staticmethod
    def qpsk_raw(freq, fs, num_samples):
        sps = 8 
        num_symbols = num_samples // sps
        symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        bits = np.random.randint(0, 4, num_symbols)
        iq_upsampled = np.repeat(symbols[bits], sps)
        return iq_upsampled[:num_samples]

    @staticmethod
    def qpsk_filtered(freq, fs, num_samples):
        sps = 8 
        num_symbols = num_samples // sps
        symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        bits = np.random.randint(0, 4, num_symbols)
        iq_upsampled = np.repeat(symbols[bits], sps)
        kernel = np.ones(sps) / sps
        return np.convolve(iq_upsampled, kernel, mode='same')[:num_samples]

    generators = {
        "Sinus": sinus,
        "Carré": carre,
        "Triangle": triangle,
        "Scie": scie,
        "Bruit": bruit,
        "QPSK_Raw": qpsk_raw,
        "QPSK_Filtered": qpsk_filtered,
    }

    @classmethod
    def generate(cls, sig_type, freq, fs, num_samples=4096):
        generator = cls.generators.get(sig_type)
        if generator:
            return generator(freq, fs, num_samples)
        return np.zeros(num_samples)