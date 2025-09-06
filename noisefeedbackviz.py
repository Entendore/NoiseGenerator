# --------------------------------------------------
# 4.  3-D waterfall → live piano (MIDI out + visual sparks)
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter1d
import mido                   # pip install mido
import threading

# --------------------------------------------------
# 4a. config
# --------------------------------------------------
SR = 44_100
CHUNK = 4_096
OVERLAP = 3_072
N_FFT = 4_096
N_MEL = 128
history = 90          # 3-D slices visible
FPS = 30
DURATION_SEC = 999    # run forever

USE_MIDI = True       # << set False to mute MIDI and keep visuals only
MIDI_PORT = None      # auto-detect
CHANNEL = 1
VELOCITY = 80

# --------------------------------------------------
# 4b. mel matrix (your fixed version)
# --------------------------------------------------
def mel_matrix(n_mel, n_fft, sr):
    mel_min, mel_max = 2595*np.log10(1+0), 2595*np.log10(1+sr/2)
    mel_pts = np.linspace(mel_min, mel_max, n_mel+2)
    hz_pts = 700*(10**(mel_pts/2595)-1)
    bin_pts = np.floor((n_fft+1)*hz_pts/sr).astype(int)
    n_bins = n_fft//2 + 1
    bin_pts = np.clip(bin_pts, 0, n_bins-1)
    mat = np.zeros((n_mel, n_bins))
    for i in range(1, n_mel+1):
        left, centre, right = bin_pts[i-1], bin_pts[i], bin_pts[i+1]
        for j in range(left, centre):
            mat[i-1, j] = (j-left)/(centre-left)
        for j in range(centre, right):
            mat[i-1, j] = (right-j)/(right-centre)
    return mat

mel_mat = mel_matrix(N_MEL, N_FFT, SR)

# --------------------------------------------------
# 4c. MIDI output thread (non-blocking)
# --------------------------------------------------
if USE_MIDI:
    try:
        MIDI_PORT = mido.open_output(mido.get_output_names()[0])
    except Exception:
        USE_MIDI = False
        print('No MIDI port found – running visual-only')

note_on = lambda n: MIDI_PORT.send(mido.Message('note_on',  channel=CHANNEL, note=n, velocity=VELOCITY)) if USE_MIDI else None
note_off = lambda n: MIDI_PORT.send(mido.Message('note_off', channel=CHANNEL, note=n)) if USE_MIDI else None

# --------------------------------------------------
# 4d. audio buffer (circular)
# --------------------------------------------------
buf_len = CHUNK*3
audio_buf = np.zeros(buf_len, dtype=float)
write_ptr = 0
gain = 1.0
SMOOTH = 0.92
TARGET_MAX = 0.65

def fill_buffer():
    global write_ptr
    # white noise + bass bump
    chunk = np.random.randn(CHUNK)*0.5
    chunk += 0.3*np.sin(2*np.pi*80*np.arange(CHUNK)/SR)*np.random.rand()
    chunk = np.tanh(chunk*0.7)
    idx = np.arange(write_ptr, write_ptr+CHUNK) % buf_len
    audio_buf[idx] = chunk
    write_ptr = (write_ptr+CHUNK) % buf_len

def adaptive_gain(col):
    global gain
    peak = np.max(col)
    desired = TARGET_MAX / (peak + 1e-8)
    gain = gain*SMOOTH + desired*(1-SMOOTH)
    return col * gain

def mel_spec(sig):
    f, t, S = spectrogram(sig, SR, window='hann', nperseg=N_FFT,
                          noverlap=OVERLAP, nfft=N_FFT, mode='magnitude')
    mel = mel_mat @ S
    return np.log10(mel + 1e-6)

# --------------------------------------------------
# 4e. 3-D grid
# --------------------------------------------------
x = np.arange(N_MEL)
y = np.arange(history)
X, Y = np.meshgrid(x, y)
spec_img = np.zeros((history, N_MEL))  # newest at row 0

# --------------------------------------------------
# 4f. piano mapping – each mel bin → MIDI note
# --------------------------------------------------
NOTE_MIN = 36   # C2
NOTE_MAX = 96   # C7
note_map = np.linspace(NOTE_MIN, NOTE_MAX, N_MEL, dtype=int)

# --------------------------------------------------
# 4g. trigger plane – cross = note-on
# --------------------------------------------------
trigger_z = -2.5
playing = set()  # currently held notes

def scan_triggers(old, new):
    """Detect rising edge across trigger plane."""
    on = (old < trigger_z) & (new >= trigger_z)
    off = (old >= trigger_z) & (new < trigger_z)
    for freq_bin in np.where(on)[0]:
        n = note_map[freq_bin]
        if n not in playing:
            note_on(n)
            playing.add(n)
    for freq_bin in np.where(off)[0]:
        n = note_map[freq_bin]
        if n in playing:
            note_off(n)
            playing.discard(n)

# --------------------------------------------------
# 4h. matplotlib 3-D setup
# --------------------------------------------------
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((4, 3, 2))
ax.set_xlim(0, N_MEL)
ax.set_ylim(0, history)
ax.set_zlim(-4, 0.5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=35, azim=-60)

# surface + wire (glow)
norm = plt.Normalize(vmin=-5, vmax=0)
surf = ax.plot_surface(X, Y, spec_img, cmap='inferno', norm=norm,
                       rstride=1, cstride=1, antialiased=False, shade=False)
wire = ax.plot_wireframe(X, Y, spec_img, color='cyan', linewidth=0.3,
                         alpha=0.6, rstride=3, cstride=3)

# trigger plane visual (semi-transparent)
trigger_plane = ax.plot_surface(X, Y, np.full_like(spec_img, trigger_z),
                                color='white', alpha=0.07, rstride=5, cstride=5)

# --------------------------------------------------
# 4i. animation
# --------------------------------------------------
def animate(i):
    global spec_img
    # ---- audio ----
    fill_buffer()
    end = write_ptr
    start = (end - CHUNK) % buf_len
    sig = np.concatenate([audio_buf[start:], audio_buf[:end]]) if start > end else audio_buf[start:end]
    col = mel_spec(sig)[:, -1]
    col = adaptive_gain(col)

    # scroll 3-D buffer (newest slice at row 0)
    old_row = spec_img[0, :].copy()
    spec_img = np.roll(spec_img, 1, axis=0)
    spec_img[0, :] = col

    # piano trigger
    scan_triggers(old_row, col)

    # ---- redraw ----
    ax.clear()
    ax.set_box_aspect((4, 3, 2))
    ax.set_xlim(0, N_MEL); ax.set_ylim(0, history); ax.set_zlim(-4, 0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=35, azim=-60)

    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, spec_img, cmap='inferno', norm=norm,
                           rstride=1, cstride=1, antialiased=False, shade=False)
    wire = ax.plot_wireframe(X, Y, spec_img, color='cyan', linewidth=0.3,
                             alpha=0.6, rstride=3, cstride=3)
    trig = ax.plot_surface(X, Y, np.full_like(spec_img, trigger_z),
                           color='white', alpha=0.07, rstride=5, cstride=5)
    return [surf, wire, trig]

ani = FuncAnimation(fig, animate, interval=1000//FPS, blit=False,
                    frames=int(DURATION_SEC*FPS), repeat=False)
plt.show()