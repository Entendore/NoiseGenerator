#!/usr/bin/env python3
"""
noise_gui.py

PyQt6 application to generate colored noise, play it, save WAV, show live 2D animation,
and export the animation to MP4 (requires ffmpeg).

Features:
 - Noise types: white, pink (Voss or 1/f), brownian, blue, violet, and arbitrary 1/f^beta
 - Beta slider for 1/f^beta (works when "Custom (beta)" chosen)
 - Device selection for playback
 - Play / Stop, Save WAV, Export MP4
 - Embedded matplotlib showing:
     * top: rolling 2D noise field
     * middle: rolling waveform
     * bottom: short-time spectrogram (higher-quality STFT)
"""

import sys
import os
import threading
import warnings
from functools import partial

import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ------------------ Configuration / Utilities ------------------
SR_DEFAULT = 44100
DURATION_DEFAULT = 6.0
AMP_DEFAULT = 0.3
GRID_H = 64
GRID_W = 128

BETA_PRESETS = {
    "White (β=0)": 0.0,
    "Pink (β=1)": 1.0,
    "Brownian (β=2)": 2.0,
    "Blue (β=-1)": -1.0,
    "Violet (β=-2)": -2.0,
    "Voss (approx pink, time-domain)": None,
    "Custom (β)": "custom"
}

_rng = np.random.default_rng()

def colored_noise_beta(beta, N, sr=SR_DEFAULT):
    """Frequency-domain colored noise generator with PSD ~ 1/f^beta."""
    # Use rfft domain shaping
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    # prevent zero division
    if len(freqs) > 1:
        freqs[0] = freqs[1]
    else:
        freqs[0] = 1.0
    re = _rng.normal(size=len(freqs))
    im = _rng.normal(size=len(freqs))
    coeff = re + 1j * im
    scale = freqs ** (-beta / 2.0)
    coeff *= scale
    sig = np.fft.irfft(coeff, n=N)
    sig -= np.mean(sig)
    m = np.max(np.abs(sig))
    if m > 0:
        sig /= m
    return sig

def voss_pink(N, rows=16):
    """Simple Voss-McCartney pink noise (approx)."""
    arr = np.zeros(N)
    rows_vals = np.zeros(rows)
    for i in range(N):
        k = 0
        while (i % (2 ** k) == 0) and (k < rows):
            rows_vals[k] = _rng.uniform(-1, 1)
            k += 1
        arr[i] = rows_vals.sum()
    arr -= np.mean(arr)
    m = np.max(np.abs(arr))
    if m > 0:
        arr /= m
    return arr

def make_noise(kind_label, duration, sr, amp, custom_beta=None, stereo=False):
    N = int(sr * duration)
    if kind_label == "Voss (approx pink, time-domain)":
        mono = voss_pink(N, rows=16)
    else:
        if BETA_PRESETS.get(kind_label) == "custom":
            beta = float(custom_beta if custom_beta is not None else 1.0)
        else:
            beta = BETA_PRESETS[kind_label]
        mono = colored_noise_beta(beta, N, sr=sr)
    mono *= float(amp)
    if stereo:
        return np.column_stack([mono, mono])
    return mono

# ------------------ PyQt6 GUI + Matplotlib Visualization ------------------

class NoiseApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Colored Noise Generator & Visualizer")
        self.setMinimumSize(1100, 700)

        # State
        self.sr = SR_DEFAULT
        self.duration = DURATION_DEFAULT
        self.amp = AMP_DEFAULT
        self.stereo = False
        self.current_signal = None
        self.play_thread = None
        self.play_event = threading.Event()
        self.stream = None

        # Build UI
        self.build_controls()
        self.build_canvas()
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.controls_layout)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Animation variables
        self.grid_h = GRID_H
        self.grid_w = GRID_W
        self.animation = None

        # Populate devices
        self.populate_devices()

    def build_controls(self):
        self.controls_layout = QtWidgets.QHBoxLayout()

        # Left column: noise settings
        left = QtWidgets.QFormLayout()
        self.kind_combo = QtWidgets.QComboBox()
        for k in BETA_PRESETS.keys():
            self.kind_combo.addItem(k)
        self.kind_combo.setCurrentText("Pink (β=1)")
        self.kind_combo.currentTextChanged.connect(self.kind_changed)
        left.addRow("Noise type:", self.kind_combo)

        self.beta_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.beta_slider.setMinimum(-200)   # maps to -2.00
        self.beta_slider.setMaximum(400)    # maps to 4.00
        self.beta_slider.setValue(100)      # 1.00
        self.beta_slider.setTickInterval(50)
        self.beta_slider.setEnabled(False)  # only for custom
        self.beta_slider.valueChanged.connect(self.beta_changed)
        self.beta_label = QtWidgets.QLabel("β = 1.00")
        beta_hbox = QtWidgets.QHBoxLayout()
        beta_hbox.addWidget(self.beta_slider)
        beta_hbox.addWidget(self.beta_label)
        left.addRow("Custom β slider:", beta_hbox)

        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 300.0)
        self.duration_spin.setValue(self.duration)
        self.duration_spin.setSingleStep(0.5)
        left.addRow("Duration (s):", self.duration_spin)

        self.amp_spin = QtWidgets.QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1.0)
        self.amp_spin.setSingleStep(0.01)
        self.amp_spin.setValue(self.amp)
        left.addRow("Amplitude:", self.amp_spin)

        self.stereo_checkbox = QtWidgets.QCheckBox("Stereo (duplicate)")
        left.addRow(self.stereo_checkbox)

        self.controls_layout.addLayout(left)

        # Middle column: device + actions
        mid = QtWidgets.QVBoxLayout()
        self.device_combo = QtWidgets.QComboBox()
        mid.addWidget(QtWidgets.QLabel("Output device:"))
        mid.addWidget(self.device_combo)

        btn_h = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_h.addWidget(self.play_btn)
        btn_h.addWidget(self.stop_btn)
        mid.addLayout(btn_h)

        self.save_btn = QtWidgets.QPushButton("Save WAV...")
        self.export_btn = QtWidgets.QPushButton("Export MP4...")
        mid.addWidget(self.save_btn)
        mid.addWidget(self.export_btn)

        self.controls_layout.addLayout(mid)

        # Right column: status / quick actions
        right = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready.")
        right.addWidget(self.status_label)
        right.addStretch()
        self.controls_layout.addLayout(right)

        # Connect buttons
        self.play_btn.clicked.connect(self.on_play)
        self.stop_btn.clicked.connect(self.on_stop)
        self.save_btn.clicked.connect(self.on_save)
        self.export_btn.clicked.connect(self.on_export)

    def populate_devices(self):
        self.device_combo.clear()
        try:
            devs = sd.query_devices()
            default_ind = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            for i, d in enumerate(devs):
                # Include only output-capable devices
                if d['max_output_channels'] > 0:
                    label = f"{i}: {d['name']} ({d['max_output_channels']} ch)"
                    self.device_combo.addItem(label, i)
                    if i == default_ind:
                        self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception as e:
            self.device_combo.addItem("Default device")
            self.status_label.setText(f"Device query failed: {e}")

    def build_canvas(self):
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.2], hspace=0.3)
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_wf = self.fig.add_subplot(gs[1])
        self.ax_spec = self.fig.add_subplot(gs[2])

        # Initial data
        self.grid = np.zeros((self.grid_h, self.grid_w))
        self.wave_buf = np.zeros(self.grid_w * 2)
        self.spec_buf = np.zeros((self.grid_h, self.grid_w))

        self.im = self.ax_img.imshow(self.grid, vmin=-1, vmax=1, aspect='auto', origin='lower')
        self.ax_img.set_title("2D noise field")
        self.ax_img.axis('off')

        self.wf_line, = self.ax_wf.plot(self.wave_buf)
        self.ax_wf.set_ylim(-1.05, 1.05)
        self.ax_wf.set_xlim(0, self.wave_buf.size)
        self.ax_wf.set_title("Rolling waveform")
        self.ax_wf.axis('off')

        self.spec_im = self.ax_spec.imshow(self.spec_buf, vmin=0, vmax=1, aspect='auto', origin='lower')
        self.ax_spec.set_title("Spectrogram (STFT)")
        self.ax_spec.axis('off')

    def kind_changed(self, txt):
        # enable beta slider only for custom
        if BETA_PRESETS.get(txt) == "custom":
            self.beta_slider.setEnabled(True)
        else:
            self.beta_slider.setEnabled(False)

    def beta_changed(self, val):
        beta = val / 100.0
        self.beta_label.setText(f"β = {beta:.2f}")

    def generate_current_signal(self):
        kind = self.kind_combo.currentText()
        dur = float(self.duration_spin.value())
        sr = int(self.sr)
        amp = float(self.amp_spin.value())
        stereo = self.stereo_checkbox.isChecked()
        custom_beta = self.beta_slider.value() / 100.0
        sig = make_noise(kind, dur, sr, amp, custom_beta=custom_beta, stereo=stereo)
        self.current_signal = sig
        return sig

    # ---------------- Playback ----------------
    def on_play(self):
        # generate
        sig = self.generate_current_signal()
        if sig is None:
            return
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Playing...")
        # set device
        dev_index = None
        try:
            dev_index = self.device_combo.currentData()
        except Exception:
            dev_index = None

        def playback_thread(sig, device):
            try:
                # If stereo present, set channels accordingly
                channels = sig.shape[1] if sig.ndim == 2 else 1
                # non-blocking callback
                idx = 0
                blocksize = 1024

                def callback(outdata, frames, time, status):
                    nonlocal idx
                    if status:
                        # show once
                        print("Playback status:", status, file=sys.stderr)
                    chunk = sig[idx:idx + frames]
                    if chunk.shape[0] < frames:
                        if sig.ndim == 1:
                            outdata[:chunk.shape[0], 0] = chunk
                            outdata[chunk.shape[0]:] = 0
                        else:
                            outdata[:chunk.shape[0], :] = chunk
                            outdata[chunk.shape[0]:] = np.zeros((frames - chunk.shape[0], channels))
                        raise sd.CallbackStop()
                    outdata[:] = chunk
                    idx += frames

                # start stream
                with sd.OutputStream(samplerate=self.sr, device=device, channels=channels,
                                     blocksize=blocksize, callback=callback):
                    # also kick off animation (on main thread via timer)
                    # run until callback finishes
                    self.start_animation(sig, self.sr)
            except Exception as e:
                self.status_label.setText(f"Playback failed: {e}")
            finally:
                # ensure UI updated on main thread
                QtCore.QMetaObject.invokeMethod(self, "on_playback_finished", QtCore.Qt.ConnectionType.QueuedConnection)

        self.play_thread = threading.Thread(target=playback_thread, args=(sig, dev_index), daemon=True)
        self.play_thread.start()

    @QtCore.pyqtSlot()
    def on_playback_finished(self):
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Ready.")
        self.stop_animation()

    def on_stop(self):
        # stopping the OutputStream is handled by callback finishing naturally; we can set a flag to stop sooner
        # For simplicity, we just stop the animation and disable play.
        # Note: advanced interruption would require a custom callback that checks a shared flag.
        self.status_label.setText("Stopping...")
        self.stop_animation()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped.")

    # ---------------- Saving / Export ----------------
    def on_save(self):
        if self.current_signal is None:
            self.generate_current_signal()
        sig = self.current_signal
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WAV", "noise.wav", "WAV files (*.wav)")
        if not fname:
            return
        try:
            sf.write(fname, sig.astype(np.float32), self.sr)
            self.status_label.setText(f"Saved WAV: {fname}")
        except Exception as e:
            self.status_label.setText(f"Save failed: {e}")

    def on_export(self):
        # Export the animation to MP4 using ffmpeg writer
        if self.current_signal is None:
            self.generate_current_signal()
        sig = self.current_signal
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MP4", "noise_anim.mp4", "MP4 files (*.mp4)")
        if not fname:
            return

        # Prepare writer
        writer = FFMpegWriter(fps=25)
        metadata = dict(title='Noise animation', artist='noise_gui')
        try:
            # Prepare a temporary short animation generation using the same update function used for live play
            frames = int(np.ceil(sig.shape[0] / max(256, int(self.sr / 25))))
            self.status_label.setText(f"Exporting MP4 ({frames} frames)... (requires ffmpeg in PATH)")
            # Because FuncAnimation stores internal references, we'll create a fresh small animation for export
            ani = FuncAnimation(self.fig, partial(self._update_frame_for_export, sig), frames=frames, interval=40, blit=False)
            ani.save(fname, writer=writer, dpi=150, metadata=metadata)
            self.status_label.setText(f"Exported MP4: {fname}")
        except Exception as e:
            self.status_label.setText(f"Export failed: {e}")
        finally:
            # re-render a final canvas
            self.canvas.draw_idle()

    # ---------------- Animation (live) ----------------
    def start_animation(self, signal_data, sr):
        # Ensure we're on the main thread for matplotlib animation
        # Build mono visualization array
        if signal_data.ndim == 2:
            mono = signal_data[:, 0]
        else:
            mono = signal_data

        self.anim_sig = mono
        self.anim_sr = sr
        # Reset buffers
        self.grid = np.zeros((self.grid_h, self.grid_w))
        self.wave_buf = np.zeros(self.grid_w * 2)
        self.spec_buf = np.zeros((self.grid_h, self.grid_w))

        # If there was an existing animation, stop it
        if self.animation:
            try:
                self.animation.event_source.stop()
            except Exception:
                pass
            self.animation = None

        # create new animation
        self.frames_total = int(np.ceil(len(mono) / max(256, int(sr / 25))))
        self.animation = FuncAnimation(self.fig, self._update_frame, frames=self.frames_total, interval=40, blit=False)
        # draw each frame via canvas draw. For embedded canvas, we need to start a timer to step animation.
        self.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)  # tiny loop to make canvas interactive

    def stop_animation(self):
        if self.animation:
            try:
                self.animation.event_source.stop()
            except Exception:
                pass
            self.animation = None

    def _update_frame(self, frame_idx):
        """Update function for live animation (called by FuncAnimation)."""
        mono = self.anim_sig
        sr = self.anim_sr
        fps = 25
        samples_per_frame = max(256, int(sr / fps))
        start = frame_idx * samples_per_frame
        end = start + samples_per_frame
        chunk = mono[start:end]
        if chunk.size == 0:
            return self.im, self.wf_line, self.spec_im

        # column for grid via downsample/interp
        col = np.interp(np.linspace(0, max(0, len(chunk)-1), self.grid_h), np.arange(max(1, len(chunk))), chunk)
        self.grid = np.roll(self.grid, -1, axis=1)
        self.grid[:, -1] = col

        # waveform buffer
        self.wave_buf = np.roll(self.wave_buf, -len(chunk))
        if len(chunk) < self.wave_buf.size:
            self.wave_buf[-len(chunk):] = chunk
        else:
            self.wave_buf[:] = chunk[-self.wave_buf.size:]

        # Better STFT: compute spectrogram of the chunk and reduce to grid_h bins
        if chunk.size >= 16:
            f, t, Sxx = signal.spectrogram(chunk, fs=sr, nperseg=256, noverlap=192, window='hann', scaling='spectrum')
            # take mean across time to collapse to frequency bins, then map to grid_h
            mags = Sxx.mean(axis=1)
            # reduce or interpolate to grid_h
            idxs = np.linspace(0, len(mags)-1, self.grid_h)
            spec_col = np.interp(idxs, np.arange(len(mags)), mags)
            if spec_col.max() > 0:
                spec_col = spec_col / spec_col.max()
        else:
            spec_col = np.zeros(self.grid_h)

        self.spec_buf = np.roll(self.spec_buf, -1, axis=1)
        self.spec_buf[:, -1] = spec_col

        # update artists
        self.im.set_data(self.grid)
        self.wf_line.set_ydata(self.wave_buf)
        self.spec_im.set_data(self.spec_buf)
        self.canvas.draw_idle()
        return self.im, self.wf_line, self.spec_im

    def _update_frame_for_export(self, sig, frame_idx):
        """Similar to _update_frame but doesn't rely on self.anim_sig; used for MP4 export FuncAnimation."""
        mono = sig if sig.ndim == 1 else sig[:, 0]
        sr = self.sr
        fps = 25
        samples_per_frame = max(256, int(sr / fps))
        start = frame_idx * samples_per_frame
        end = start + samples_per_frame
        chunk = mono[start:end]
        if chunk.size == 0:
            return self.im, self.wf_line, self.spec_im

        col = np.interp(np.linspace(0, max(0, len(chunk)-1), self.grid_h), np.arange(max(1, len(chunk))), chunk)
        self.grid = np.roll(self.grid, -1, axis=1)
        self.grid[:, -1] = col

        self.wave_buf = np.roll(self.wave_buf, -len(chunk))
        if len(chunk) < self.wave_buf.size:
            self.wave_buf[-len(chunk):] = chunk
        else:
            self.wave_buf[:] = chunk[-self.wave_buf.size:]

        if chunk.size >= 16:
            f, t, Sxx = signal.spectrogram(chunk, fs=sr, nperseg=256, noverlap=192, window='hann', scaling='spectrum')
            mags = Sxx.mean(axis=1)
            idxs = np.linspace(0, len(mags)-1, self.grid_h)
            spec_col = np.interp(idxs, np.arange(len(mags)), mags)
            if spec_col.max() > 0:
                spec_col = spec_col / spec_col.max()
        else:
            spec_col = np.zeros(self.grid_h)

        self.spec_buf = np.roll(self.spec_buf, -1, axis=1)
        self.spec_buf[:, -1] = spec_col

        self.im.set_data(self.grid)
        self.wf_line.set_ydata(self.wave_buf)
        self.spec_im.set_data(self.spec_buf)
        return self.im, self.wf_line, self.spec_im

# ------------------ main ------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = NoiseApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
