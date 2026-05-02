import numpy as np

# Audio Settings
SR_DEFAULT = 44100
CHUNK_SIZE = 1024

# Video Settings
VIDEO_FPS = 30

# Color Palettes
PALETTES = {
    "Cyberpunk": {
        'wave': (0, 255, 255),
        'waterfall': (255, 0, 255),
        'rain_base': np.array([0.0, 1.0, 1.0]),
        'rain_mod': np.array([1.0, 0.0, 1.0]),
        'spectral': 'cool'
    },
    "Ocean": {
        'wave': (0, 200, 255),
        'waterfall': (0, 100, 255),
        'rain_base': np.array([0.0, 0.2, 0.8]),
        'rain_mod': np.array([0.0, 0.8, 0.8]),
        'spectral': 'ocean'
    },
    "Inferno": {
        'wave': (255, 100, 0),
        'waterfall': (255, 50, 0),
        'rain_base': np.array([1.0, 0.5, 0.0]),
        'rain_mod': np.array([1.0, 0.0, 0.0]),
        'spectral': 'warm'
    },
    "Matrix": {
        'wave': (0, 255, 0),
        'waterfall': (0, 255, 100),
        'rain_base': np.array([0.0, 1.0, 0.0]),
        'rain_mod': np.array([0.0, 0.5, 0.0]),
        'spectral': 'greens'
    }
}