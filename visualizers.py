import math
import numpy as np
from scipy import signal
from scipy.fft import rfft
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt

# Check for MIDI support
try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

class WaveformWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buffer = np.zeros(2048)
        self.pen_color = (0, 255, 255)
        
    def set_palette(self, palette):
        self.pen_color = palette.get('wave', (0, 255, 255))
        
    def set_data(self, data):
        if data.ndim == 2: data = data[:, 0]
        self.buffer = np.roll(self.buffer, -len(data))
        self.buffer[-len(data):] = data
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter(self)
        qp.fillRect(self.rect(), Qt.black)
        qp.setPen(QtGui.QPen(QtGui.QColor(*self.pen_color), 1))
        w, h = self.width(), self.height()
        center = h / 2
        step = max(1, len(self.buffer) / w)
        
        path = QtGui.QPainterPath()
        path.moveTo(0, center - self.buffer[0] * center)
        for i in range(w):
            idx = int(i * step)
            y = center - self.buffer[idx] * center
            path.lineTo(i, y)
        qp.drawPath(path)

class SpectrogramWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.h = 129
        self.w = 200
        self.img_data = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.palette_type = 'cool'
        
    def set_palette(self, palette):
        self.palette_type = palette.get('spectral', 'cool')
        
    def set_data(self, data, sr):
        if data.ndim == 2: data = data[:, 0]
        f, t, Sxx = signal.spectrogram(data, fs=sr, nperseg=256, noverlap=192)
        Sxx_db = 10 * np.log10(Sxx + 1e-6)
        
        norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-6)
        
        if norm.shape[0] != self.h:
            norm = np.interp(np.linspace(0, 1, self.h), np.linspace(0, 1, norm.shape[0]), norm[:, -1])
            col = (norm * 255).astype(np.uint8)
        else:
            col = (norm[:, -1] * 255).astype(np.uint8)
            
        self.img_data = np.roll(self.img_data, -1, axis=1)
        
        if self.palette_type == 'cool':
            self.img_data[:, -1, 0] = col 
            self.img_data[:, -1, 1] = np.clip(col * 0.5, 0, 255) 
            self.img_data[:, -1, 2] = np.clip(255 - col, 0, 255) 
        elif self.palette_type == 'warm':
            self.img_data[:, -1, 0] = np.clip(255 - col, 0, 255) 
            self.img_data[:, -1, 1] = np.clip(col * 0.5, 0, 255) 
            self.img_data[:, -1, 2] = col 
        elif self.palette_type == 'ocean':
            self.img_data[:, -1, 0] = 0
            self.img_data[:, -1, 1] = np.clip(col * 0.5, 0, 255)
            self.img_data[:, -1, 2] = col
        elif self.palette_type == 'greens':
            self.img_data[:, -1, 0] = 0
            self.img_data[:, -1, 1] = col
            self.img_data[:, -1, 2] = 0
        else: 
            self.img_data[:, -1, 0] = col
            self.img_data[:, -1, 1] = col
            self.img_data[:, -1, 2] = col

        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter(self)
        qimg = QtGui.QImage(self.img_data.data, self.w, self.h, self.w * 3, QtGui.QImage.Format_RGB888)
        qp.drawImage(self.rect(), qimg)

class WaterfallWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = 40
        self.n_bins = 64
        self.grid = np.zeros((self.history, self.n_bins))
        self.line_color = (0, 255, 255)
        self.midi_out = None
        if MIDI_AVAILABLE:
            try:
                names = mido.get_output_names()
                if names: self.midi_out = mido.open_output(names[0])
            except: pass
        self.note_map = np.linspace(36, 84, self.n_bins, dtype=int)
        self.prev_col = np.zeros(self.n_bins)

    def set_palette(self, palette):
        self.line_color = palette.get('waterfall', (255, 0, 255))

    def set_data(self, data, sr):
        if data.ndim == 2: data = data[:, 0]
        N = len(data)
        yf = rfft(data)
        mag = np.abs(yf[:N//2])
        
        col = np.interp(np.linspace(0, 1, self.n_bins), np.linspace(0, 1, len(mag)), mag)
        col = (col - col.min()) / (col.max() - col.min() + 1e-6)
        
        self.grid = np.roll(self.grid, -1, axis=0)
        self.grid[-1, :] = col
        
        if self.midi_out:
            for i, val in enumerate(col):
                n = self.note_map[i]
                if val > 0.8 and self.prev_col[i] <= 0.8:
                    self.midi_out.send(mido.Message('note_on', note=n, velocity=100))
                elif val <= 0.8 and self.prev_col[i] > 0.8:
                    self.midi_out.send(mido.Message('note_off', note=n))
            self.prev_col = col
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter(self)
        qp.fillRect(self.rect(), Qt.black)
        w, h = self.width(), self.height()
        cx, cy = w/2, h/2
        
        for i in range(self.history):
            alpha = int(255 * (i / self.history))
            color = QtGui.QColor(*self.line_color, alpha)
            qp.setPen(QtGui.QPen(color, 1))
            
            row = self.grid[i]
            path = QtGui.QPainterPath()
            
            for j in range(self.n_bins):
                val = row[j]
                x_3d = (j - self.n_bins/2) * 4
                z_3d = (self.history - i) * 2
                y_3d = val * 50
                
                angle = 0.5
                y_proj = (y_3d * math.cos(angle) - z_3d * math.sin(angle))
                z_proj = (y_3d * math.sin(angle) + z_3d * math.cos(angle))
                
                factor = 1.0 + z_proj / 200.0
                screen_x = cx + x_3d * factor
                screen_y = cy - y_proj * factor
                
                if j == 0: path.moveTo(screen_x, screen_y)
                else: path.lineTo(screen_x, screen_y)
            qp.drawPath(path)

class RaindropsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.size = 128
        self.u = np.zeros((self.size, self.size))
        self.col_base = np.array([0.0, 1.0, 1.0])
        self.col_mod = np.array([1.0, 0.0, 1.0])

    def set_palette(self, palette):
        self.col_base = palette.get('rain_base', np.array([0.0, 1.0, 1.0]))
        self.col_mod = palette.get('rain_mod', np.array([1.0, 0.0, 1.0]))

    def set_data(self, data, sr):
        if data.ndim == 2: data = data[:, 0]
        rms = np.sqrt(np.mean(data**2))
        
        if np.random.rand() < rms * 5:
            x, y = np.random.randint(10, self.size-10, 2)
            r = int(5 + rms * 10)
            Y, X = np.ogrid[-r:r, -r:r]
            mask = X**2 + Y**2 <= r**2
            self.u[x-r:x+r, y-r:y+r][mask] += 1.0
        
        lap = (-4*self.u + np.roll(self.u,1,axis=0) + np.roll(self.u,-1,axis=0) +
               np.roll(self.u,1,axis=1) + np.roll(self.u,-1,axis=1))
        self.u += lap * 0.25
        self.u *= 0.98
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter(self)
        img_norm = np.clip(self.u, -1, 1)
        
        r = (self.col_base[0] * (np.sin(img_norm * 3.14) + 1) * 127 + self.col_mod[0] * (np.cos(img_norm * 1.5) + 1) * 60).astype(np.uint8)
        g = (self.col_base[1] * (np.sin(img_norm * 3.14) + 1) * 127 + self.col_mod[1] * (np.cos(img_norm * 1.5) + 1) * 60).astype(np.uint8)
        b = (self.col_base[2] * (np.sin(img_norm * 3.14) + 1) * 127 + self.col_mod[2] * (np.cos(img_norm * 1.5) + 1) * 60).astype(np.uint8)

        rgb = np.stack([r, g, b, np.full_like(r, 255)], axis=-1)
        qimg = QtGui.QImage(rgb.data, self.size, self.size, self.size * 4, QtGui.QImage.Format_RGBA8888)
        qp.drawImage(self.rect(), qimg)

class PsychedelicWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.size = 64
        self.hue = np.random.rand(self.size, self.size)
        self.val = np.zeros((self.size, self.size))
        
    def set_palette(self, palette):
        pass 

    def hsv_to_rgb(self, h, s, v):
        c = v * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = v - c
        h_int = (h * 6).astype(int) % 6
        r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
        
        mask0 = (h_int == 0); r[mask0]=c[mask0]; g[mask0]=x[mask0]
        mask1 = (h_int == 1); r[mask1]=x[mask1]; g[mask1]=c[mask1]
        mask2 = (h_int == 2); g[mask2]=c[mask2]; b[mask2]=x[mask2]
        mask3 = (h_int == 3); g[mask3]=x[mask3]; b[mask3]=c[mask3]
        mask4 = (h_int == 4); r[mask4]=x[mask4]; b[mask4]=c[mask4]
        mask5 = (h_int == 5); r[mask5]=c[mask5]; b[mask5]=x[mask5]
        
        return ((r+m)*255).astype(np.uint8), ((g+m)*255).astype(np.uint8), ((b+m)*255).astype(np.uint8)

    def set_data(self, data, sr):
        if data.ndim == 2: data = data[:, 0]
        fft = np.abs(rfft(data))
        bass = np.mean(fft[1:5])
        
        shift = int(1 + bass/100)
        self.hue = np.roll(self.hue, shift, axis=0)
        self.val = np.roll(self.val, -shift, axis=1)
        
        self.val += np.random.randn(self.size, self.size) * 0.05
        self.hue += np.random.randn(self.size, self.size) * 0.01
        self.val = np.clip(self.val, 0, 1)
        self.hue = self.hue % 1.0
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter(self)
        r, g, b = self.hsv_to_rgb(self.hue, np.ones((self.size, self.size))*0.8, self.val)
        rgb = np.stack([r, g, b, np.full_like(r, 255)], axis=-1)
        qimg = QtGui.QImage(rgb.data, self.size, self.size, self.size * 4, QtGui.QImage.Format_RGBA8888)
        qp.drawImage(self.rect(), qimg)