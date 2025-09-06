import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
grid_size = 256
time_steps = 500
layers = 4
fps = 30

# Noise and music influence per layer
noise_scale = [0.04, 0.03, 0.02, 0.01]
music_scale = [3.0, 2.5, 2.0, 1.5]

# Ripple damping factor
damping = 0.95

# -----------------------------
# Generate smooth procedural noise
# -----------------------------
def generate_noise(shape, scale=0.05):
    noise = np.random.randn(*shape)
    noise = np.cumsum(noise, axis=0)
    noise = np.cumsum(noise, axis=1)
    noise = noise - np.min(noise)
    noise = noise / np.max(noise)
    return noise * scale

# -----------------------------
# Generate procedural music wave
# -----------------------------
def generate_music_wave(t, size):
    freqs = np.linspace(1, 20, 6)  # multiple sine frequencies
    wave = np.zeros(size)
    for f in freqs:
        wave += np.sin(2 * np.pi * f * t / fps + np.random.rand() * np.pi)
    wave /= len(freqs)
    return wave

# -----------------------------
# Initialize figure
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')
data = np.zeros((grid_size, grid_size))
img = ax.imshow(data, cmap='hsv', interpolation='bilinear', vmin=0, vmax=1)

# For ripple effect across layers
layer_buffers = [np.zeros((grid_size, grid_size)) for _ in range(layers)]

# -----------------------------
# Animation update function
# -----------------------------
def update(frame):
    global layer_buffers
    combined = np.zeros((grid_size, grid_size))
    
    for l in range(layers):
        # Add procedural noise
        noise = generate_noise((grid_size, grid_size), scale=noise_scale[l])
        
        # Add music influence
        music = generate_music_wave(frame, grid_size)
        music_2d = np.tile(music, (grid_size, 1))
        layer_buffers[l] = layer_buffers[l] * damping + noise + music_2d * music_scale[l]
        
        combined += layer_buffers[l]
    
    # Normalize per-frame
    combined = combined - np.min(combined)
    combined = combined / np.max(combined)
    
    img.set_data(combined)
    return [img]

# -----------------------------
# Run animation
# -----------------------------
ani = FuncAnimation(fig, update, frames=time_steps, interval=1000/fps, blit=True)
plt.show()
