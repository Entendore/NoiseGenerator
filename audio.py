import sounddevice as sd
from PySide6 import QtCore
import numpy as np
from config import CHUNK_SIZE

class AudioManager(QtCore.QObject):
    data_ready = QtCore.Signal(np.ndarray)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.stream = None
        self.is_playing = False

    def start(self):
        if self.is_playing: return
        self.engine.reset_state()
        self.is_playing = True
        try:
            self.stream = sd.OutputStream(
                samplerate=self.engine.sr, channels=2 if self.engine.stereo else 1,
                callback=self._audio_callback, blocksize=CHUNK_SIZE, dtype=np.float32
            )
            self.stream.start()
        except Exception as e: print(f"Audio Error: {e}")

    def stop(self):
        self.is_playing = False
        if self.stream: self.stream.stop(); self.stream.close(); self.stream = None

    def _audio_callback(self, outdata, frames, time, status):
        if not self.is_playing: outdata.fill(0); return
        data = self.engine.get_block(frames)
        outdata[:] = data.astype(np.float32)
        self.data_ready.emit(data.copy())