import numpy as np
from scipy import signal
from config import SR_DEFAULT

class NoiseEngine:
    def __init__(self, sr=SR_DEFAULT):
        self.sr = sr
        self.type = "Pink (β=1)"
        self.amplitude = 0.5
        self.stereo = False
        self.custom_beta = 1.0
        
        self._brown_state = 0.0
        self._pink_state = np.zeros(16)
        self._pink_key = 0
        self._perlin_pos = 0
        
        self._perm = np.arange(256, dtype=int)
        np.random.shuffle(self._perm)
        self._perm = np.concatenate([self._perm, self._perm])

    def reset_state(self):
        self._brown_state = 0.0
        self._pink_state = np.zeros(16)
        self._pink_key = 0
        self._perlin_pos = 0

    def _generate_chunk_freq_domain(self, n_samples, beta):
        freqs = np.fft.rfftfreq(n_samples, d=1.0/self.sr)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        amps = freqs ** (-beta / 2.0) if beta != 0 else np.ones_like(freqs)
        amps[np.isinf(amps)] = 0
        
        spectrum = amps * (np.cos(phases) + 1j * np.sin(phases))
        wave = np.fft.irfft(spectrum, n=n_samples)
        peak = np.max(np.abs(wave))
        return (wave / peak) if peak > 0 else wave

    def _generate_pink_voss(self, n_samples):
        rows = 16
        out = np.zeros(n_samples)
        state = self._pink_state
        key = self._pink_key
        
        for i in range(n_samples):
            key += 1
            if key >= 2**rows: key = 0
            if key == 0: state = np.random.uniform(-1, 1, rows)
            else:
                k = key; idx = 0
                while k & 1: idx += 1; k >>= 1
                if idx < rows: state[idx] = np.random.uniform(-1, 1)
            out[i] = state.sum()
        
        self._pink_state = state
        self._pink_key = key
        return out / (rows / 2.0)

    def _generate_brown(self, n_samples):
        out = np.zeros(n_samples)
        state = self._brown_state
        for i in range(n_samples):
            state = 0.995 * state + 0.02 * np.random.randn()
            out[i] = state
        self._brown_state = state
        peak = np.max(np.abs(out))
        return (out / peak) if peak > 0 else out

    def _generate_perlin(self, n_samples, freq=5.0):
        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        def grad(h): return 1 if (h & 1) == 0 else -1
        out = np.zeros(n_samples)
        for i in range(n_samples):
            x = (self._perlin_pos + i) * freq / self.sr
            xi = int(x) & 255
            xf = x - int(x)
            u = fade(xf)
            p = self._perm
            out[i] = (grad(p[p[xi]]) * (1-u) + grad(p[p[xi+1]]) * u)
        self._perlin_pos += n_samples
        return out * 2.0

    def get_block(self, n_samples):
        t = self.type
        if "White" in t: wave = np.random.randn(n_samples)
        elif "Pink" in t and "Voss" not in t: wave = self._generate_chunk_freq_domain(n_samples, 1.0)
        elif "Voss" in t: wave = self._generate_pink_voss(n_samples)
        elif "Brown" in t: wave = self._generate_brown(n_samples)
        elif "Blue" in t: wave = self._generate_chunk_freq_domain(n_samples, -1.0)
        elif "Violet" in t: wave = self._generate_chunk_freq_domain(n_samples, -2.0)
        elif "Custom" in t: wave = self._generate_chunk_freq_domain(n_samples, self.custom_beta)
        elif "Perlin" in t: wave = self._generate_perlin(n_samples)
        elif "Band" in t:
            w = np.random.randn(n_samples)
            b, a = signal.butter(4, [200, 8000], btype='bandpass', fs=self.sr)
            wave = signal.lfilter(b, a, w)
            m = np.max(np.abs(wave))
            if m > 0: wave /= m
        else: wave = np.random.randn(n_samples)

        wave *= self.amplitude
        if self.stereo: return np.column_stack([wave, wave])
        return wave.reshape(-1, 1)