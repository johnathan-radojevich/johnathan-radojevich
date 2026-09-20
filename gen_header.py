"""
Generates an animated GIF header using the Gray-Scott reaction-diffusion model.
Produces a wide banner (1200x300) suitable for a GitHub README.
"""
import numpy as np
from PIL import Image
import os

# --- Grid & simulation parameters ---
W, H = 200, 100          # simulation grid
SCALE = 4                # pixel upscale factor
OUT_W, OUT_H = 800, 200  # final banner size

Du, Dv = 0.14, 0.06
F, k   = 0.035, 0.065   # "coral" / spot-forming regime

dt     = 1.0
WARMUP = 8000            # steps before capturing frames
FRAMES = 40
STEPS_PER_FRAME = 15

rng = np.random.default_rng(42)

# --- Initialization ---
U = np.ones((H, W), dtype=np.float64)
V = np.zeros((H, W), dtype=np.float64)

# Multiple random seeds scattered across the grid
for _ in range(30):
    cx = rng.integers(10, W - 10)
    cy = rng.integers(10, H - 10)
    r  = rng.integers(4, 10)
    U[cy-r:cy+r, cx-r:cx+r] = 0.50 + 0.02 * rng.standard_normal((2*r, 2*r))
    V[cy-r:cy+r, cx-r:cx+r] = 0.25 + 0.02 * rng.standard_normal((2*r, 2*r))

def lap(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4.0 * Z)

def step(U, V, n=1):
    for _ in range(n):
        uvv   = U * V * V
        U    += dt * (Du * lap(U) - uvv + F * (1.0 - U))
        V    += dt * (Dv * lap(V) + uvv - (F + k) * V)
        U[:] = np.clip(U, 0.0, 1.0)
        V[:] = np.clip(V, 0.0, 1.0)
    return U, V

print("Running warmup…")
U, V = step(U, V, WARMUP)

# --- Colormap: viridis-like (blue→green→yellow), hand-coded via lookup ---
def apply_colormap(arr):
    """Map [0,1] float array to RGB uint8 using a viridis-like palette."""
    # viridis sampled at 6 stops (R,G,B in 0-255)
    stops = np.array([
        [68,   1,  84],
        [59,  82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
        [253, 231, 37],
    ], dtype=np.float32)
    n = len(stops) - 1
    idx   = np.clip(arr * n, 0, n - 1e-9)
    lo    = idx.astype(int)
    frac  = (idx - lo)[..., np.newaxis]
    rgb   = stops[lo] * (1.0 - frac) + stops[lo + 1] * frac
    return rgb.astype(np.uint8)

print("Capturing frames…")
frames = []
for i in range(FRAMES):
    U, V = step(U, V, STEPS_PER_FRAME)
    # V field shows the "activator" pattern most clearly
    arr = V.copy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    rgb = apply_colormap(arr)                     # H x W x 3
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((OUT_W, OUT_W // 2), Image.NEAREST)  # 800x400
    # Crop to banner height
    top = (img.height - OUT_H) // 2
    img = img.crop((0, top, OUT_W, top + OUT_H))
    frames.append(img)
    print(f"  frame {i+1}/{FRAMES}", end="\r", flush=True)

print()

out = os.path.join(os.path.dirname(__file__), "header-anim.gif")
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=100,       # ms per frame  (~10 fps)
    optimize=False,
)
print(f"Saved → {out}")
