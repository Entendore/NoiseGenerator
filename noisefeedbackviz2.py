import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter, rotate
from scipy.fft import rfft, rfftfreq

# ---------------- Parameters ----------------
sample_rate = 44100
chunk_size = 1024
duration_seconds = 12
frame_interval = 30
freq_bins = 256  # Increased resolution for more detail
total_frames = 300

layers = 12  # More layers for richer visuals
base_diff_rates = np.linspace(0.06, 0.18, layers)
F_base = 0.03
DT = 0.25
DAMPING = 0.992  # Slightly less damping for longer-lasting effects

# ---------------- Initialize RD layers ----------------
RD_layers = []
hue_offsets = np.linspace(0, 1, layers, endpoint=False)
for i in range(layers):
    RD_layers.append({
        'value': np.clip(0.5 + 0.02*np.random.randn(freq_bins, freq_bins), 0, 1),
        'vx': np.zeros((freq_bins, freq_bins)),
        'vy': np.zeros((freq_bins, freq_bins)),
        'wave': np.zeros((freq_bins, freq_bins)),
        'wave_prev': np.zeros((freq_bins, freq_bins)),
        'hue': hue_offsets[i],  # More organized color distribution
        'hue_shift': np.random.rand() * 0.1  # Individual layer hue variation
    })

# ---------------- Particles ----------------
textures = ['rain', 'wind', 'laughter', 'white', 'sparkle']  # Added sparkle texture
particles_layers = {tex: [] for tex in textures}

# ---------------- Noise Functions ----------------
def noise_rain(n): 
    return np.random.normal(0, 0.5, n)

def noise_wind(n): 
    return gaussian_filter(np.random.randn(n), sigma=5)

def noise_laughter(n):
    x = np.random.randn(n)
    mod = np.sin(np.linspace(0, 10*np.pi, n) * np.random.rand()) * 0.5 + 0.5
    return x * mod

def noise_white(n): 
    return np.random.normal(0, 1, n)

def noise_sparkle(n):
    # Sparkle noise with occasional sharp peaks
    base = np.random.normal(0, 0.3, n)
    spikes = (np.random.rand(n) > 0.98) * np.random.rand(n) * 5
    return base + spikes

texture_funcs = {
    'rain': noise_rain,
    'wind': noise_wind,
    'laughter': noise_laughter,
    'white': noise_white,
    'sparkle': noise_sparkle
}

# ---------------- Laplacian & Advection ----------------
def laplacian(Z):
    return (-4*Z + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1))

def advect(Z, vx, vy):
    coords_x, coords_y = np.meshgrid(np.arange(Z.shape[0]), np.arange(Z.shape[1]), indexing='ij')
    x_new = np.clip(coords_x - vx, 0, Z.shape[0]-1)
    y_new = np.clip(coords_y - vy, 0, Z.shape[1]-1)
    x0, y0 = x_new.astype(int), y_new.astype(int)
    return Z[x0, y0]

# ---------------- Enhanced Visualization Functions ----------------
def create_vignette(shape, intensity=0.7):
    """Create a vignette effect to draw focus to the center"""
    rows, cols = shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    vignette = 1 - np.sqrt(X**2 + Y**2) * intensity
    return np.clip(vignette, 0, 1)[:, :, np.newaxis]

def create_radial_gradient(shape):
    """Create a radial gradient for more interesting backgrounds"""
    center_x, center_y = shape[0] // 2, shape[1] // 2
    y, x = np.ogrid[:shape[0], :shape[1]]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    gradient = 1 - distance / np.max(distance)
    return gradient

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_facecolor('black')
rgb_img = np.zeros((freq_bins, freq_bins, 3))
img_plot = ax.imshow(rgb_img, origin='lower', interpolation='bicubic')  # Better interpolation
ax.axis('off')

# Precompute visual effects
vignette = create_vignette((freq_bins, freq_bins), intensity=0.5)
radial_gradient = create_radial_gradient((freq_bins, freq_bins))

# ---------------- Animation Update ----------------
def update(frame_idx):
    global RD_layers, particles_layers

    combined_value = np.zeros((freq_bins, freq_bins))
    combined_hue = np.zeros((freq_bins, freq_bins))
    particle_img = np.zeros((freq_bins, freq_bins))
    wave_pattern = np.zeros((freq_bins, freq_bins))

    # Global time-varying parameters
    time_factor = frame_idx * 0.01
    global_hue_shift = 0.05 * np.sin(time_factor * 0.5)
    
    # ---------------- Visual → Noise Feedback ----------------
    layer_activity = np.array([np.mean(layer['value']) for layer in RD_layers])
    feedback_factor = 0.5 + 0.5 * np.sin(time_factor + layer_activity.sum() * 2 * np.pi)
    feedback_factor *= 0.8 + 0.4 * np.random.rand()
    texture_noise_amplitude = {tex: feedback_factor * np.random.uniform(0.5, 1.2) for tex in textures}

    for tex in textures:
        # Procedural noise modulated by feedback and particle density
        chunk = texture_funcs[tex](chunk_size) * texture_noise_amplitude[tex]
        fft_vals = np.abs(rfft(chunk))
        fft_freqs = rfftfreq(len(chunk), 1/sample_rate)
        
        # More detailed frequency analysis
        bass = np.mean(fft_vals[(fft_freqs > 20) & (fft_freqs < 150)])
        low_mids = np.mean(fft_vals[(fft_freqs > 150) & (fft_freqs < 500)])
        high_mids = np.mean(fft_vals[(fft_freqs > 500) & (fft_freqs < 2000)])
        highs = np.mean(fft_vals[(fft_freqs > 2000) & (fft_freqs < 8000)])
        
        norm_bass = np.clip(bass/500, 0, 2)
        norm_low_mids = np.clip(low_mids/400, 0, 2)
        norm_high_mids = np.clip(high_mids/300, 0, 2)
        norm_highs = np.clip(highs/200, 0, 2)
        
        diff_rates_mod = base_diff_rates * (1 + 0.5 * norm_low_mids)
        F_mod = F_base * (1 + norm_bass * 0.7)

        for layer_idx, layer in enumerate(RD_layers):
            # Dynamic hue shifting
            layer_hue = (layer['hue'] + global_hue_shift + 
                         layer['hue_shift'] * np.sin(time_factor * 0.2 + layer_idx * 0.5)) % 1.0
            
            # Random droplet injection with more varied sizes
            if np.random.rand() < norm_bass * 0.2:
                x, y = np.random.randint(10, freq_bins-10, 2)
                r = int(np.random.randint(3, 15) * (1 + norm_bass))
                Y, X = np.ogrid[-r:r, -r:r]
                mask = X**2 + Y**2 <= r**2
                val = np.linspace(0.2, 0.8, mask.sum())
                layer['value'][x-r:x+r, y-r:y+r][mask] = val
                layer['wave'][x-r:x+r, y-r:y+r][mask] += 0.3 * (1 + norm_bass)
                
                # Add velocity disturbance
                angle = np.random.rand() * 2 * np.pi
                strength = np.random.uniform(0.3, 1.0)
                layer['vx'][x-r:x+r, y-r:y+r][mask] += strength * np.sin(angle)
                layer['vy'][x-r:x+r, y-r:y+r][mask] += strength * np.cos(angle)
                
                # Add particles with more properties
                if tex == 'sparkle':
                    # Sparkles are smaller and brighter
                    for _ in range(int(5 + norm_bass * 10)):
                        particles_layers[tex].append([
                            x + np.random.uniform(-r, r), 
                            y + np.random.uniform(-r, r),
                            np.random.uniform(0.2, 0.6),
                            np.random.uniform(0, 2*np.pi),
                            1.0,  # Initial brightness
                            np.random.rand(),  # Hue
                            np.random.uniform(0.5, 2.0)  # Size multiplier
                        ])
                else:
                    particles_layers[tex].append([
                        x, y, 
                        np.random.uniform(0.3, 0.8), 
                        np.random.uniform(-np.pi, np.pi), 
                        1.0,  # Initial brightness
                        layer_hue  # Use layer hue for cohesion
                    ])

            # Wave propagation with frequency-dependent damping
            wave_damping = DAMPING * (1 - 0.1 * norm_highs)
            wave_new = (2 * layer['wave'] - layer['wave_prev'] + 
                        laplacian(layer['wave']) * 0.6) * wave_damping
            layer['wave_prev'] = layer['wave'].copy()
            layer['wave'] = wave_new

            # Accumulate wave pattern for visualization
            wave_pattern += layer['wave'] * (1.0 / layers)

            # Turbulent reaction-diffusion
            L = laplacian(layer['value'])
            dV = (diff_rates_mod[layer_idx] * L - 
                  layer['value']**3 + F_mod * (1 - layer['value']))
            layer['value'] += (dV * DT + 
                              0.03 * np.random.randn(freq_bins, freq_bins) * norm_highs)
            layer['value'] = np.clip(layer['value'], 0, 1)

            # Advection + turbulence with frequency influence
            turbulence = norm_high_mids * 0.03
            layer['vx'] += np.random.randn(freq_bins, freq_bins) * turbulence
            layer['vy'] += np.random.randn(freq_bins, freq_bins) * turbulence
            layer['value'] = advect(layer['value'], layer['vx'], layer['vy'])
            layer['vx'] *= 0.91  # More persistence in velocity
            layer['vy'] *= 0.91

            # Accumulate value and hue with drifting
            combined_value += (layer['value'] + 
                              norm_highs * 0.3 * np.sin(layer['wave'] * 8 + time_factor * 2 + layer_idx))
            combined_hue += layer_hue / layers

    # Enhanced particle system
    for tex in textures:
        new_particles = []
        for p in particles_layers[tex]:
            if tex == 'sparkle':
                row, col, speed, angle, bright, hue_val, size = p
                # Sparkles move in more random patterns
                angle += np.random.uniform(-0.2, 0.2)
                row += speed * np.sin(angle)
                col += speed * np.cos(angle)
                speed *= 0.98  # Sparkles maintain speed longer
                bright *= 0.99  # Sparkles fade slower
                
                if 0 <= row < freq_bins and 0 <= col < freq_bins and bright > 0.05:
                    new_particles.append([row, col, speed, angle, bright, hue_val, size])
                    # Draw a brighter, larger trail for sparkles
                    r = int(2 * size)
                    for i in range(-r, r+1):
                        for j in range(-r, r+1):
                            if i*i + j*j <= r*r:
                                r_idx = int(np.clip(row + i, 0, freq_bins-1))
                                c_idx = int(np.clip(col + j, 0, freq_bins-1))
                                # Distance-based brightness falloff
                                dist = np.sqrt(i*i + j*j) / r
                                particle_img[r_idx, c_idx] += bright * (1 - dist) * 8
            else:
                row, col, speed, angle, bright, hue_val = p
                row += speed * np.sin(angle + row * 0.05)
                col += speed * np.cos(angle + col * 0.02)
                speed *= 0.94
                bright *= 0.96
                
                if 0 <= row < freq_bins and 0 <= col < freq_bins and bright > 0.05:
                    new_particles.append([row, col, speed, angle, bright, hue_val])
                    # Draw a small trail with color variation
                    for dx, dy in [(-1,0), (0,-1), (1,0), (0,1), (0,0)]:
                        r_idx = int(np.clip(row + dx, 0, freq_bins-1))
                        c_idx = int(np.clip(col + dy, 0, freq_bins-1))
                        particle_img[r_idx, c_idx] += bright * 6
                        
                        # Add a touch of the particle's hue to the combined_hue
                        if bright > 0.2:
                            combined_hue[r_idx, c_idx] = (combined_hue[r_idx, c_idx] + hue_val * 0.1) / 1.1
        
        particles_layers[tex] = new_particles

    # Add wave pattern to the value for more organic feel
    combined_value = combined_value * (1 + 0.3 * wave_pattern)
    
    # HSV → RGB with enhanced color processing
    hue_smoothed = gaussian_filter(combined_hue, sigma=1.2)
    
    # More dynamic saturation - varies across space and time
    saturation = np.clip(0.6 + 0.4 * np.sin(time_factor + radial_gradient * 2 * np.pi), 0.5, 1.0)
    
    # Value processing with more contrast
    value_blurred = gaussian_filter(np.clip(combined_value / np.max(combined_value + 1e-12), 0, 1), sigma=1.0)
    value_enhanced = np.power(value_blurred, 0.7)  # Gamma correction for better contrast
    value_enhanced = value_enhanced * (1 + 2.0 * particle_img / (1 + particle_img))  # Blend particles
    
    # Apply radial gradient and vignette
    value_enhanced = value_enhanced * radial_gradient
    value_enhanced = np.clip(value_enhanced, 0, 1)
    
    # Create HSV image
    hsv = np.stack([hue_smoothed, saturation, value_enhanced], axis=2)
    rgb_img = hsv_to_rgb(hsv)
    
    # Apply vignette effect
    rgb_img = rgb_img * vignette
    
    # Subtle color balance adjustment
    rgb_img[:, :, 0] = np.clip(rgb_img[:, :, 0] * 1.1, 0, 1)  # Boost reds slightly
    rgb_img[:, :, 2] = np.clip(rgb_img[:, :, 2] * 0.95, 0, 1)  # Reduce blues slightly
    
    img_plot.set_data(rgb_img)

    return [img_plot]

# ---------------- Run Animation ----------------
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=frame_interval, blit=False)
plt.tight_layout(pad=0)
plt.show()