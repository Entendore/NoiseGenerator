import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal, Slot, QThreadPool, QTimer

from config import PALETTES, VIDEO_FPS
from engine import NoiseEngine
from audio import AudioManager
from workers import WavExportWorker, VideoWriterThread
from visualizers import WaveformWidget, SpectrogramWidget, WaterfallWidget, RaindropsWidget, PsychedelicWidget

class NoiseStudio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Noise Studio (Structured)")
        self.resize(1000, 700)
        
        self.thread_pool = QThreadPool()
        
        self.engine = NoiseEngine()
        self.audio_manager = AudioManager(self.engine)
        self.audio_manager.data_ready.connect(self.on_audio_data)
        
        # Recording State
        self.recorder_thread = None
        self.is_recording = False
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.stop_record)
        
        # Video Frame Capture Timer
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.capture_video_frame)
        
        # Store current recording resolution
        self.rec_width = 640
        self.rec_height = 480
        
        self.setup_ui()
        
        self.param_timer = QTimer()
        self.param_timer.timeout.connect(self.update_params)
        self.param_timer.start(50)
        
        self.apply_palette("Cyberpunk")

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        
        # Left
        left = QtWidgets.QVBoxLayout()
        
        # --- Engine ---
        grp_noise = QtWidgets.QGroupBox("Engine")
        form = QtWidgets.QFormLayout(grp_noise)
        
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItems(["White", "Pink (β=1)", "Brown (β=2)", "Blue (β=-1)", 
                                  "Violet (β=-2)", "Custom (β)", "Voss Pink", "Perlin", "Band"])
        self.combo_type.currentTextChanged.connect(lambda t: self.slider_beta.setEnabled("Custom" in t))
        form.addRow("Type:", self.combo_type)
        
        hb = QtWidgets.QHBoxLayout()
        self.slider_beta = QtWidgets.QSlider(Qt.Horizontal); self.slider_beta.setRange(-200, 400); self.slider_beta.setValue(100); self.slider_beta.setEnabled(False)
        self.lbl_beta = QtWidgets.QLabel("1.00")
        self.slider_beta.valueChanged.connect(lambda v: self.lbl_beta.setText(f"{v/100:.2f}"))
        hb.addWidget(self.slider_beta); hb.addWidget(self.lbl_beta)
        form.addRow("Beta:", hb)
        
        self.spin_amp = QtWidgets.QDoubleSpinBox(); self.spin_amp.setRange(0, 1); self.spin_amp.setValue(0.5); self.spin_amp.setSingleStep(0.05)
        form.addRow("Amp:", self.spin_amp)
        self.chk_stereo = QtWidgets.QCheckBox("Stereo"); form.addRow(self.chk_stereo)
        
        self.combo_palette = QtWidgets.QComboBox()
        self.combo_palette.addItems(PALETTES.keys())
        self.combo_palette.currentTextChanged.connect(self.apply_palette)
        form.addRow("Palette:", self.combo_palette)
        
        left.addWidget(grp_noise)
        
        # --- Transport ---
        grp_play = QtWidgets.QGroupBox("Transport")
        hp = QtWidgets.QHBoxLayout(grp_play)
        self.btn_play = QtWidgets.QPushButton("▶ Play"); self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_play.clicked.connect(self.start_audio); self.btn_stop.clicked.connect(self.stop_audio)
        hp.addWidget(self.btn_play); hp.addWidget(self.btn_stop)
        left.addWidget(grp_play)
        
        # --- Recording ---
        grp_rec = QtWidgets.QGroupBox("Recording (FFMPEG)")
        hv = QtWidgets.QVBoxLayout(grp_rec)
        
        # Resolution Selector
        h_res = QtWidgets.QHBoxLayout()
        h_res.addWidget(QtWidgets.QLabel("Resolution:"))
        self.combo_resolution = QtWidgets.QComboBox()
        self.combo_resolution.addItems([
            "Small (640x360)", 
            "YouTube (1920x1080)", 
            "Shorts (1080x1920)"
        ])
        h_res.addWidget(self.combo_resolution)
        hv.addLayout(h_res)

        h_dur = QtWidgets.QHBoxLayout()
        h_dur.addWidget(QtWidgets.QLabel("Duration (s):"))
        self.spin_rec_duration = QtWidgets.QDoubleSpinBox()
        self.spin_rec_duration.setRange(0, 3600)
        self.spin_rec_duration.setValue(0)
        self.spin_rec_duration.setSpecialValueText("Manual")
        h_dur.addWidget(self.spin_rec_duration)
        hv.addLayout(h_dur)
        
        h_btns = QtWidgets.QHBoxLayout()
        self.btn_start_rec = QtWidgets.QPushButton("🔴 Start Record")
        self.btn_stop_rec = QtWidgets.QPushButton("⏹ Stop Record")
        self.btn_stop_rec.setEnabled(False)
        
        self.btn_start_rec.clicked.connect(self.start_record)
        self.btn_stop_rec.clicked.connect(self.stop_record)
        
        h_btns.addWidget(self.btn_start_rec)
        h_btns.addWidget(self.btn_stop_rec)
        hv.addLayout(h_btns)
        
        # Status Label
        self.lbl_rec_status = QtWidgets.QLabel("Ready")
        self.lbl_rec_status.setStyleSheet("color: gray;")
        hv.addWidget(self.lbl_rec_status)
        
        # Progress Bar for Recording (Hidden initially)
        self.rec_progress_bar = QtWidgets.QProgressBar()
        self.rec_progress_bar.setVisible(False)
        hv.addWidget(self.rec_progress_bar)
        
        left.addWidget(grp_rec)
        
        # --- WAV Export ---
        grp_exp = QtWidgets.QGroupBox("WAV Export")
        he = QtWidgets.QVBoxLayout(grp_exp)
        
        h_wav = QtWidgets.QHBoxLayout()
        self.spin_wav_dur = QtWidgets.QDoubleSpinBox()
        self.spin_wav_dur.setRange(1, 600)
        self.spin_wav_dur.setValue(10)
        h_wav.addWidget(QtWidgets.QLabel("Dur:"))
        h_wav.addWidget(self.spin_wav_dur)
        he.addLayout(h_wav)
        
        self.btn_save = QtWidgets.QPushButton("Save WAV (Background)")
        self.btn_save.clicked.connect(self.save_wav)
        he.addWidget(self.btn_save)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        he.addWidget(self.progress_bar)
        
        left.addWidget(grp_exp)
        
        left.addStretch()
        
        # --- Center ---
        self.tabs = QtWidgets.QTabWidget()
        
        tab_std = QtWidgets.QWidget(); lay_std = QtWidgets.QVBoxLayout(tab_std)
        self.viz_wave = WaveformWidget()
        self.viz_spec = SpectrogramWidget()
        lay_std.addWidget(self.viz_wave, 1)
        lay_std.addWidget(self.viz_spec, 2)
        
        self.viz_water = WaterfallWidget()
        self.viz_rain = RaindropsWidget()
        self.viz_psy = PsychedelicWidget()
        
        self.tabs.addTab(tab_std, "Standard")
        self.tabs.addTab(self.viz_water, "3D Waterfall")
        self.tabs.addTab(self.viz_rain, "Raindrops")
        self.tabs.addTab(self.viz_psy, "Psychedelic")
        
        layout.addLayout(left, 1)
        layout.addWidget(self.tabs, 3)

    def apply_palette(self, name):
        p = PALETTES.get(name, PALETTES["Cyberpunk"])
        self.viz_wave.set_palette(p)
        self.viz_spec.set_palette(p)
        self.viz_water.set_palette(p)
        self.viz_rain.set_palette(p)
        self.viz_psy.set_palette(p)

    def update_params(self):
        self.engine.type = self.combo_type.currentText()
        self.engine.custom_beta = self.slider_beta.value() / 100.0
        self.engine.amplitude = self.spin_amp.value()
        self.engine.stereo = self.chk_stereo.isChecked()

    def start_audio(self): self.audio_manager.start()
    def stop_audio(self): self.audio_manager.stop()

    @Slot(np.ndarray)
    def on_audio_data(self, data):
        self.viz_wave.set_data(data)
        
        c = self.tabs.currentWidget()
        if c == self.tabs.widget(0): self.viz_spec.set_data(data, self.engine.sr)
        elif c == self.viz_water: self.viz_water.set_data(data, self.engine.sr)
        elif c == self.viz_rain: self.viz_rain.set_data(data, self.engine.sr)
        elif c == self.viz_psy: self.viz_psy.set_data(data, self.engine.sr)

    # --- RECORDING LOGIC ---
    
    def start_record(self):
        if not self.audio_manager.is_playing:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please start audio playback before recording.")
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Video", "noise.mp4", "MP4 (*.mp4)")
        if not fname: return

        # Determine resolution based on selection
        res_text = self.combo_resolution.currentText()
        if "Small" in res_text:
            width, height = 640, 360
        elif "Shorts" in res_text:
            width, height = 1080, 1920
        else: # YouTube
            width, height = 1920, 1080
            
        self.rec_width = width
        self.rec_height = height
        
        duration = self.spin_rec_duration.value()
        
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(VIDEO_FPS),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            fname
        ]
        
        # Pass duration to worker for progress calculation
        self.recorder_thread = VideoWriterThread(cmd, width, height, duration)
        self.recorder_thread.finished_signal.connect(self.on_recording_finished)
        self.recorder_thread.progress_signal.connect(self.on_rec_progress)
        self.recorder_thread.start()

        self.is_recording = True
        self.btn_start_rec.setEnabled(False)
        self.btn_stop_rec.setEnabled(True)
        
        self.lbl_rec_status.setText("Recording...")
        self.lbl_rec_status.setStyleSheet("color: red; font-weight: bold;")
        
        self.video_timer.start(int(1000 / VIDEO_FPS))
        
        if duration > 0:
            self.record_timer.start(int(duration * 1000))
            self.lbl_rec_status.setText(f"Rec {duration}s...")
            self.rec_progress_bar.setVisible(True)
            self.rec_progress_bar.setValue(0)

    @Slot(int)
    def on_rec_progress(self, percent):
        if percent > 0: # Only show bar if we have meaningful progress (finalizing)
            if not self.rec_progress_bar.isVisible():
                self.rec_progress_bar.setVisible(True)
            self.rec_progress_bar.setValue(percent)
            self.lbl_rec_status.setText(f"Finalizing... {percent}%")

    def stop_record(self):
        if self.record_timer.isActive():
            self.record_timer.stop()
        
        self.video_timer.stop()
        self.is_recording = False
            
        if self.recorder_thread:
            self.recorder_thread.stop()
            
        self.lbl_rec_status.setText("Finalizing...")
        self.lbl_rec_status.setStyleSheet("color: orange; font-weight: bold;")

    def on_recording_finished(self):
        if self.recorder_thread:
            self.recorder_thread.deleteLater()
            self.recorder_thread = None
            
        self.btn_start_rec.setEnabled(True)
        self.btn_stop_rec.setEnabled(False)
        self.rec_progress_bar.setVisible(False)
        
        self.lbl_rec_status.setText("Saved.")
        self.lbl_rec_status.setStyleSheet("color: green;")
        self.statusBar().showMessage("Recording finished.")

    def capture_video_frame(self):
        if not self.recorder_thread or not self.is_recording:
            return
            
        width, height = self.rec_width, self.rec_height
        target = self.tabs.currentWidget()
        
        qimg = target.grab().toImage()
        qimg = qimg.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
        
        ptr = qimg.bits()
        total_bytes = qimg.sizeInBytes()
        data = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes)
        
        if data.size == width * height * 3:
            data = data.reshape((height, width, 3))
            self.recorder_thread.add_frame(data.tobytes())

    # --- WAV EXPORT ---

    def save_wav(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WAV", "noise.wav", "WAV (*.wav)")
        if not fname: return
        
        duration = self.spin_wav_dur.value()
        
        params = {
            'sr': 44100, 'type': self.engine.type, 'custom_beta': self.engine.custom_beta,
            'amplitude': self.engine.amplitude, 'stereo': self.engine.stereo
        }
        
        worker = WavExportWorker(params, duration, fname)
        worker.signals.finished.connect(self.on_wav_finished)
        worker.signals.error.connect(self.on_wav_error)
        worker.signals.progress.connect(self.on_wav_progress)
        
        self.btn_save.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Exporting WAV ({duration}s)...")
        
        self.thread_pool.start(worker)

    @Slot(int)
    def on_wav_progress(self, val): self.progress_bar.setValue(val)

    @Slot(str)
    def on_wav_finished(self, fname):
        self.btn_save.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Saved {fname}")
        QtWidgets.QMessageBox.information(self, "Done", f"Saved {fname}")

    @Slot(str)
    def on_wav_error(self, err):
        self.btn_save.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Error", f"Export failed: {err}")

    def closeEvent(self, e):
        self.stop_record()
        self.stop_audio()
        if hasattr(self.viz_water, 'midi_out') and self.viz_water.midi_out:
            self.viz_water.midi_out.close()
        super().closeEvent(e)