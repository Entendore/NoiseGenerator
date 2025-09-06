#!/usr/bin/env python3
"""
Qt6 Noise Generator (PyQt6 version)
Dependencies:
    pip install PyQt6 numpy sounddevice pyqtgraph scipy
"""

import sys, math, wave, threading, numpy as np, sounddevice as sd
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from scipy.signal import spectrogram

################################
# DSP: Noise Generator
################################

class NoiseGenerator:
    def __init__(self, sample_rate=44100, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.type = "white"
        self.amplitude = 0.2
        # states
        self._brown_last = np.zeros(channels)
        self._init_pink()
        # params
        self.damping = 0.999
        self.perlin_freq = 5.0
        self.band_low, self.band_high = 200, 8000
        # perlin init
        self.perlin_perm = np.arange(256, dtype=int)
        np.random.shuffle(self.perlin_perm)
        self.perlin_perm = np.concatenate([self.perlin_perm, self.perlin_perm])
        self._perlin_pos = 0

    def _init_pink(self):
        self.pink_rows = 16
        self.pink_max_key = 2**self.pink_rows
        self.pink_key = 0
        self.pink_rows_vals = np.zeros((self.pink_rows + 1, self.channels))

    def set_type(self, t):
        self.type = t
        if t == "pink": self._init_pink()
        if t == "brown": self._brown_last = np.zeros(self.channels)
        if t == "perlin": self._perlin_pos = 0

    def _white(self, frames):
        return np.random.uniform(-1,1,(frames,self.channels))

    def _pink(self, frames):
        out = np.empty((frames, self.channels), dtype=np.float32)
        for i in range(frames):
            self.pink_key += 1
            if self.pink_key >= self.pink_max_key: self.pink_key = 0
            key, idx = self.pink_key, 0
            while key & 1 == 0: key >>= 1; idx += 1
            self.pink_rows_vals[idx] = np.random.uniform(-1,1,self.channels)
            out[i] = np.sum(self.pink_rows_vals, axis=0)
        return out/(self.pink_rows+1)

    def _brown(self, frames):
        out = np.empty((frames,self.channels),dtype=np.float32)
        for i in range(frames):
            w = np.random.normal(0.0,1.0,self.channels)
            self._brown_last = (self._brown_last+0.02*w)*self.damping
            out[i]=self._brown_last
        return out

    def _grad(self,h): return (h%12)/6.0-1.0
    def _perlin1d(self,frames):
        def fade(t): return 6*t**5-15*t**4+10*t**3
        out=np.zeros((frames,self.channels),dtype=np.float32)
        for i in range(frames):
            x=(self._perlin_pos+i)*self.perlin_freq/self.sample_rate
            x0=int(math.floor(x))&255; x1=(x0+1)&255
            sx=fade(x-math.floor(x))
            for ch in range(self.channels):
                g0=self._grad(self.perlin_perm[x0]); g1=self._grad(self.perlin_perm[x1])
                v0=g0*(x-math.floor(x)); v1=g1*(x-math.floor(x)-1.0)
                out[i,ch]=(1-sx)*v0+sx*v1
        self._perlin_pos+=frames
        return out*2.0

    def _blue(self,frames):
        x=np.random.randn(frames,self.channels)
        b=np.convolve(x[:,0],[1,-1],mode="same").reshape(-1,1)
        return np.hstack([b,b]) if self.channels==2 else b

    def _violet(self,frames):
        x=np.random.randn(frames,self.channels)
        b=np.convolve(x[:,0],[1,-2,1],mode="same").reshape(-1,1)
        return np.hstack([b,b]) if self.channels==2 else b

    def _bandlimited(self,frames):
        n=frames; noise=np.random.randn(n,self.channels)
        freqs=np.fft.rfftfreq(n,1.0/self.sample_rate)
        filt=((freqs>=self.band_low)&(freqs<=self.band_high)).astype(float)
        out=[]
        for ch in range(self.channels):
            spec=np.fft.rfft(noise[:,ch]); spec*=filt
            sig=np.fft.irfft(spec)
            out.append(sig)
        return np.stack(out,axis=1)

    def next_block(self,frames):
        if self.type=="white": b=self._white(frames)
        elif self.type=="pink": b=self._pink(frames)
        elif self.type=="brown": b=self._brown(frames)
        elif self.type=="perlin": b=self._perlin1d(frames)
        elif self.type=="blue": b=self._blue(frames)
        elif self.type=="violet": b=self._violet(frames)
        elif self.type=="band": b=self._bandlimited(frames)
        else: b=self._white(frames)
        return (b*self.amplitude).astype(np.float32)

################################
# Qt Widgets
################################

class WaveformWidget(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.buffer=np.zeros((1,1))
    def set_buffer(self,buf):
        if buf is None or buf.size==0: self.buffer=np.zeros((1,1))
        else: self.buffer=buf if buf.ndim==2 else buf.reshape(-1,1)
        self.update()
    def paintEvent(self,e):
        qp=QtGui.QPainter(self); r=self.rect()
        qp.fillRect(r,QtGui.QColor(26,26,26))
        if self.buffer.size==0: return
        ch=self.buffer[:,0]; w,h=r.width(),r.height()
        step=max(1,len(ch)//w); samples=ch[::step][:w]
        center,half=h//2,(h//2)-2
        path=QtGui.QPainterPath(); path.moveTo(0,center-samples[0]*half)
        for i,s in enumerate(samples):
            x=int(i*(w/len(samples))); y=center-s*half
            path.lineTo(x,y)
        qp.setPen(QtGui.QPen(QtGui.QColor(255,255,255))); qp.drawPath(path)

################################
# Main Window
################################

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Noise Generator — PyQt6")
        self.generator=NoiseGenerator(); self.sample_rate=44100
        self.channels=1; self.frames_per_buffer=1024
        self.stream=None; self.playing=False
        self.audio_lock=threading.Lock()
        self._setup_ui(); self.update_generator()
        self.timer=QtCore.QTimer(self); self.timer.setInterval(50)
        self.timer.timeout.connect(self._refresh); self.timer.start()

    def _setup_ui(self):
        c=QtWidgets.QWidget(); self.setCentralWidget(c); v=QtWidgets.QVBoxLayout(c)
        row=QtWidgets.QHBoxLayout(); v.addLayout(row)
        row.addWidget(QtWidgets.QLabel("Noise:"))
        self.combo=QtWidgets.QComboBox()
        self.combo.addItems(["white","pink","brown","perlin","blue","violet","band"])
        self.combo.currentTextChanged.connect(self.update_generator); row.addWidget(self.combo)
        row.addWidget(QtWidgets.QLabel("Amp:"))
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0,100); self.slider.setValue(20)
        self.slider.valueChanged.connect(self.update_generator); row.addWidget(self.slider)
        row.addWidget(QtWidgets.QLabel("Channels:"))
        self.combo_ch=QtWidgets.QComboBox(); self.combo_ch.addItems(["Mono","Stereo"])
        self.combo_ch.currentIndexChanged.connect(self._ch_changed); row.addWidget(self.combo_ch)
        self.btn=QtWidgets.QPushButton("Play"); self.btn.clicked.connect(self.toggle); row.addWidget(self.btn)
        self.save=QtWidgets.QPushButton("Save WAV"); self.save.clicked.connect(self.save_wav); row.addWidget(self.save)
        # Param slider
        self.param=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.param.setRange(1,100); self.param.valueChanged.connect(self.update_params)
        v.addWidget(QtWidgets.QLabel("Parameter")); v.addWidget(self.param)
        # Displays
        self.wave=WaveformWidget(); v.addWidget(self.wave)
        self.spec=pg.ImageView(); self.spec.setMinimumHeight(180); v.addWidget(self.spec)
        self.status=self.statusBar()

    def _ch_changed(self,i): self.channels=1 if i==0 else 2; self.generator.channels=self.channels; self.update_generator()
    def update_generator(self):
        self.generator.sample_rate=self.sample_rate; self.generator.channels=self.channels
        self.generator.set_type(self.combo.currentText()); self.generator.amplitude=self.slider.value()/100.0
    def update_params(self):
        val=self.param.value(); t=self.generator.type
        if t=="brown": self.generator.damping=0.990+val/10000.0
        elif t=="perlin": self.generator.perlin_freq=val/2.0
        elif t=="band": self.generator.band_high=200+val*200
    def audio_cb(self,out,frames,time,status):
        with self.audio_lock:
            block=self.generator.next_block(frames)
            out[:]=np.tile(block,(1,self.channels)) if block.shape[1]!=self.channels else block
            self._last=out.copy()
    def toggle(self): self.stop() if self.playing else self.start()
    def start(self):
        try:
            self.update_generator()
            self.stream=sd.OutputStream(samplerate=self.sample_rate,blocksize=self.frames_per_buffer,
                channels=self.channels,dtype="float32",callback=self.audio_cb)
            self.stream.start(); self.playing=True; self.btn.setText("Stop")
            self.status.showMessage("Playing")
        except Exception as e: self.status.showMessage(f"Error: {e}")
    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()
        self.stream=None; self.playing=False; self.btn.setText("Play"); self.status.showMessage("Stopped")
    def _refresh(self):
        buf=self._last if hasattr(self,"_last") else self.generator.next_block(2048)
        self.wave.set_buffer(buf)
        f,t,S= spectrogram(buf[:,0],fs=self.sample_rate,nperseg=128,noverlap=64)
        self.spec.setImage(20*np.log10(S+1e-6),autoLevels=False)
    def save_wav(self):
        fname,_=QtWidgets.QFileDialog.getSaveFileName(self,"Save WAV","noise.wav","Wave (*.wav)")
        if not fname: return
        dur,ok=QtWidgets.QInputDialog.getDouble(self,"Duration","Seconds:",5.0,0.1,600.0,1)
        if not ok: return
        frames=int(dur*self.sample_rate); wf=wave.open(fname,"wb")
        wf.setnchannels(self.channels); wf.setsampwidth(2); wf.setframerate(self.sample_rate)
        chunk=4096; written=0
        while written<frames:
            n=min(chunk,frames-written); block=self.generator.next_block(n)
            data=(np.clip(block,-1,1)*32767).astype(np.int16).tobytes()
            wf.writeframes(data); written+=n
        wf.close(); self.status.showMessage(f"Saved {fname}")
    def closeEvent(self,e): self.stop(); super().closeEvent(e)

################################
def main(argv):
    app=QtWidgets.QApplication(argv); w=MainWindow(); w.resize(900,600); w.show(); sys.exit(app.exec())
if __name__=="__main__": main(sys.argv)
