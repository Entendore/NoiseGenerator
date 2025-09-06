import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft, rfftfreq
import sounddevice as sd

# -------------------------------
# Parameters
# -------------------------------
size = 250
layers = 3
diff_rates = [0.12, 0.10, 0.08]
F, k = 0.03, 0.06
dt = 1.0
damping = 0.995
CHUNK = 1024
SAMPLE_RATE = 44100

# -------------------------------
# Shared audio buffer
# -------------------------------
audio_buffer = np.zeros(CHUNK, dtype=np.float32)
frame_counter = 0

# -------------------------------
# Procedural multi-mode noise
# -------------------------------
def generate_multi_mode_noise(length=1024, sample_rate=44100, frame=0):
    freqs = np.linspace(0, sample_rate/2, length//2+1)
    w_pleasant = 0.3 + 0.3*np.sin(frame*0.001)
    w_chaotic = 0.2 + 0.3*np.sin(frame*0.002 + 1)
    w_rhythmic = 0.2 + 0.2*np.sin(frame*0.0015 + 2)
    w_flowing = 0.3 + 0.2*np.sin(frame*0.0005 + 3)

    pleasant = sum(np.exp(-0.5*((freqs-c)/80)**2) for c in [110,220,440,880])
    
    chaotic = np.zeros(len(freqs))
    for _ in range(5):
        peak = np.random.randint(50, len(freqs)-50)
        chaotic[peak:peak+10] += np.random.rand()*2
    
    rhythmic = np.sin(2*np.pi*freqs*frame*0.0005)**2
    flowing = np.exp(-0.5*((freqs-20)/100)**2) * np.random.rand(len(freqs))
    
    spectrum = w_pleasant*pleasant + w_chaotic*chaotic + w_rhythmic*rhythmic + w_flowing*flowing
    phases = np.exp(1j*2*np.pi*np.random.rand(len(freqs)))
    full_spectrum = spectrum * phases
    mirrored = np.conjugate(full_spectrum[-2:0:-1])
    full_fft = np.concatenate([full_spectrum, mirrored])
    
    signal = np.fft.ifft(full_fft).real
    signal /= np.max(np.abs(signal)) + 1e-6
    return signal.astype(np.float32)

# -------------------------------
# Audio callback (non-blocking)
# -------------------------------
def audio_callback(outdata, frames, time, status):
    global frame_counter, audio_buffer
    audio_buffer = generate_multi_mode_noise(length=frames, sample_rate=SAMPLE_RATE, frame=frame_counter)
    outdata[:] = audio_buffer.reshape(-1,1)
    frame_counter += 1

stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK, callback=audio_callback)
stream.start()

# -------------------------------
# Initialize visual layers
# -------------------------------
R_layers = [0.5*np.ones((size,size)) for _ in range(layers)]
G_layers = [0.5*np.ones((size,size)) for _ in range(layers)]
B_layers = [0.5*np.ones((size,size)) for _ in range(layers)]
wave_layers = [np.zeros((size,size)) for _ in range(layers)]
wave_prev_layers = [np.zeros((size,size)) for _ in range(layers)]
vx_layers = [np.zeros((size,size)) for _ in range(layers)]
vy_layers = [np.zeros((size,size)) for _ in range(layers)]

# -------------------------------
# Core visual functions
# -------------------------------
def add_droplet(layer, strength=1.0):
    x, y = np.random.randint(20, size-20, 2)
    r = int(np.random.randint(10, 20) * strength)
    Y, X = np.ogrid[-r:r, -r:r]
    mask = X**2 + Y**2 <= r**2
    val = np.linspace(0.2,0.7,mask.sum())
    color_choice = np.random.choice(['R','G','B'])
    if color_choice == 'R':
        R_layers[layer][x-r:x+r, y-r:y+r][mask] = val
    elif color_choice == 'G':
        G_layers[layer][x-r:x+r, y-r:y+r][mask] = val
    else:
        B_layers[layer][x-r:x+r, y-r:y+r][mask] = val
    wave_layers[layer][x-r:x+r, y-r:y+r][mask] += strength * np.linspace(0.2,0.5,mask.sum())
    vx_layers[layer][x-r:x+r, y-r:y+r][mask] += np.random.uniform(-0.5,0.5,mask.sum())
    vy_layers[layer][x-r:x+r, y-r:y+r][mask] += np.random.uniform(-0.5,0.5,mask.sum())

def laplacian(Z):
    return (-4*Z + np.roll(Z,1,axis=0) + np.roll(Z,-1,axis=0)
            + np.roll(Z,1,axis=1) + np.roll(Z,-1,axis=1))

def advect(Z, vx, vy):
    coords_x, coords_y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_new = np.clip(coords_x - vx, 0, size-1)
    y_new = np.clip(coords_y - vy, 0, size-1)
    x0, y0 = x_new.astype(int), y_new.astype(int)
    return Z[x0, y0]

# -------------------------------
# Update visuals
# -------------------------------
def update(frame):
    fft_vals = np.abs(rfft(audio_buffer))
    freqs = rfftfreq(len(audio_buffer), 1/SAMPLE_RATE)
    bass = np.mean(fft_vals[(freqs>20)&(freqs<150)])
    mids = np.mean(fft_vals[(freqs>150)&(freqs<1000)])
    highs = np.mean(fft_vals[(freqs>1000)&(freqs<5000)])

    norm_bass = np.clip(bass/500, 0, 2)
    norm_mids = np.clip(mids/500, 0, 2)
    norm_highs = np.clip(highs/500, 0, 2)

    for layer in range(layers):
        # Bass → raindrop size & intensity
        if np.random.rand() < norm_bass*0.2:
            add_droplet(layer, strength=1 + norm_bass*2)

        # Ripple propagation
        wave_new = (2*wave_layers[layer] - wave_prev_layers[layer] + laplacian(wave_layers[layer])*0.5)*damping
        wave_prev_layers[layer] = wave_layers[layer].copy()
        wave_layers[layer] = wave_new

        # Reaction-diffusion influenced by audio
        diff_scale = 0.5 + norm_mids
        F_scaled = F * (0.5 + norm_highs)
        LR, LG, LB = laplacian(R_layers[layer]), laplacian(G_layers[layer]), laplacian(B_layers[layer])
        dR = diff_scale*LR - R_layers[layer]*G_layers[layer]*B_layers[layer] + F_scaled*(1 - R_layers[layer])
        dG = diff_scale*LG - G_layers[layer]*B_layers[layer]*R_layers[layer] + F_scaled*(1 - G_layers[layer])
        dB = diff_scale*LB - B_layers[layer]*R_layers[layer]*G_layers[layer] + F_scaled*(1 - B_layers[layer])
        # Add small noise for psychedelic effect
        R_layers[layer] += dR*dt + 0.05*np.random.randn(size,size)
        G_layers[layer] += dG*dt + 0.05*np.random.randn(size,size)
        B_layers[layer] += dB*dt + 0.05*np.random.randn(size,size)

        # Mids → swirling/advection
        vx_layers[layer] += np.random.randn(size,size) * 0.02 * norm_mids
        vy_layers[layer] += np.random.randn(size,size) * 0.02 * norm_mids
        R_layers[layer] = advect(R_layers[layer], vx_layers[layer], vy_layers[layer])
        G_layers[layer] = advect(G_layers[layer], vx_layers[layer], vy_layers[layer])
        B_layers[layer] = advect(B_layers[layer], vx_layers[layer], vy_layers[layer])
        vx_layers[layer] *= 0.95
        vy_layers[layer] *= 0.95

    # Combine layers with psychedelic modulation
    R_vis, G_vis, B_vis = np.zeros((size,size)), np.zeros((size,size)), np.zeros((size,size))
    for layer in range(layers):
        R_vis += np.clip(R_layers[layer] + np.sin(wave_layers[layer]*6 + frame*0.02 + layer)*0.3*norm_highs,0,1)
        G_vis += np.clip(G_layers[layer] + np.sin(wave_layers[layer]*7 + frame*0.025 + layer*1.5)*0.3*norm_highs,0,1)
        B_vis += np.clip(B_layers[layer] + np.sin(wave_layers[layer]*8 + frame*0.03 + layer*3.0)*0.3*norm_highs,0,1)

    rgb = np.dstack((
        np.clip(R_vis/layers,0,1),
        np.clip(G_vis/layers,0,1),
        np.clip(B_vis/layers,0,1)
    ))

    mat.set_data(rgb)
    return [mat]

# -------------------------------
# Visualization setup
# -------------------------------
fig, ax = plt.subplots(figsize=(6,6))
mat = ax.imshow(np.zeros((size,size,3)), interpolation='bilinear')
ax.axis('off')

ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
plt.show()
