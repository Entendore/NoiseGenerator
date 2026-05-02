import subprocess
import numpy as np
import soundfile as sf
import re
from queue import Queue, Empty
from PySide6 import QtCore
from config import SR_DEFAULT, VIDEO_FPS
from engine import NoiseEngine

# --- WAV Export Worker ---
class WavExportWorker(QtCore.QRunnable):
    def __init__(self, engine_params, duration, fname):
        super().__init__()
        self.params = engine_params
        self.duration = duration
        self.fname = fname
        self.signals = WorkerSignals()

    def run(self):
        try:
            eng = NoiseEngine(sr=self.params['sr'])
            eng.type = self.params['type']
            eng.custom_beta = self.params['custom_beta']
            eng.amplitude = self.params['amplitude']
            eng.stereo = self.params['stereo']
            eng.reset_state()
            
            total_samples = int(self.duration * SR_DEFAULT)
            chunk_size = 2048
            chunks = []
            steps = max(1, int(total_samples / chunk_size))
            for i in range(steps):
                block = eng.get_block(min(chunk_size, total_samples - len(chunks)*chunk_size))
                chunks.append(block)
                if i % 10 == 0: self.signals.progress.emit(int((i/steps)*100))
            
            sig = np.concatenate(chunks)
            sf.write(self.fname, sig, SR_DEFAULT)
            self.signals.finished.emit(self.fname)
        except Exception as e:
            self.signals.error.emit(str(e))

class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)

# --- Video Writer Thread ---
class VideoWriterThread(QtCore.QThread):
    finished_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int) # 0-100 progress

    def __init__(self, cmd, width, height, duration=0):
        super().__init__()
        self.cmd = cmd
        self.width = width
        self.height = height
        self.duration = duration # Expected duration in seconds
        self.queue = Queue(maxsize=60)
        self.running = True
        self.process = None

    def run(self):
        try:
            # REMOVED universal_newlines=True so stdin accepts binary data
            self.process = subprocess.Popen(
                self.cmd, 
                stdin=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"FFMPEG Error: {e}")
            self.finished_signal.emit()
            return

        # Phase 1: Write frames (Binary Mode)
        while self.running or not self.queue.empty():
            try:
                frame_data = self.queue.get(timeout=0.1)
                if self.process:
                    # Write raw bytes
                    self.process.stdin.write(frame_data)
            except Empty:
                continue
            except BrokenPipeError:
                break
        
        # Phase 2: Finalizing (Close stdin and read progress)
        if self.process:
            self.process.stdin.close()
            
            time_regex = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
            
            while True:
                # Read bytes from stderr
                line_bytes = self.process.stderr.readline()
                if not line_bytes:
                    if self.process.poll() is not None:
                        break
                    continue
                
                # Decode bytes to string manually
                line = line_bytes.decode('utf-8', errors='ignore')
                
                # Parse time from ffmpeg output
                match = time_regex.search(line)
                if match and self.duration > 0:
                    try:
                        h, m, s = map(float, match.groups())
                        current_time = h * 3600 + m * 60 + s
                        pct = int((current_time / self.duration) * 100)
                        # Cap at 99% until fully done
                        self.progress_signal.emit(min(pct, 99))
                    except ValueError:
                        pass
            
            self.process.wait()
            # Emit 100% done
            self.progress_signal.emit(100)
        
        self.finished_signal.emit()

    def add_frame(self, data):
        if not self.running: return False
        if self.queue.full(): return False
        self.queue.put(data)
        return True

    def stop(self):
        self.running = False