"""
Generates an animated GIF header using the Barkley excitable-medium model —
the reaction-diffusion system underpinning spatial epidemic wave propagation.

  u (activator / infected)   → red-orange
  v (inhibitor / recovered)  → blue-teal
  rest (susceptible)         → dark navy background

Produces a wide banner (800×200) suitable for a GitHub README.
"""
import numpy as np
from PIL import Image
import os

W, H = 800, 200
FRAMES = 60
DURATION = 80  # ms per frame

# Barkley model parameters — tuned for clean rotating spiral waves
a = 0.75
b = 0.06
eps = 0.02
D_u = 1.0   # infected diffuse; recovered/susceptible do not
dt = 0.08
WARMUP = 3000
STEPS_PER_FRAME = 12


def reaction_u(u, v):
    return (1.0 / eps) * u * (1.0 - u) * (u - (v + b) / a)


def lap2d(u):
    return (
        np.roll(u, 1, 0) + np.roll(u, -1, 0) +
        np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u
    )


def step(u, v):
    u_new = u + dt * (D_u * lap2d(u) + reaction_u(u, v))
    v_new = v + dt * (u - v)
    return np.clip(u_new, 0.0, 1.0), np.clip(v_new, 0.0, 1.0)


def colorize(u, v):
    # Susceptible: dark navy   Infected: hot red-orange   Recovered: cool blue
    r_ch = u * 0.92 + v * 0.04 + (1 - u - v) * 0.02
    g_ch = u * 0.22 + v * 0.30 + (1 - u - v) * 0.04
    b_ch = u * 0.05 + v * 0.88 + (1 - u - v) * 0.10
    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


# Initialise near the quiescent state, then plant broken-wave-front seeds
# so that free spiral ends roll up into self-sustaining rotors.
np.random.seed(17)
u = np.zeros((H, W))
v = np.zeros((H, W))

seed_xs = [100, 260, 420, 580, 720]
seed_ys = [H // 2 - 20, H // 2 + 25, H // 2 - 30, H // 2 + 20, H // 2 - 15]
radii_i = [9, 11, 10, 9, 10]
radii_r = [18, 20, 19, 18, 19]
orientations = [0.3, -0.4, 0.5, -0.3, 0.4]  # break angle (radians)

ys, xs = np.ogrid[:H, :W]
for cx, cy, ri, ro, phi in zip(seed_xs, seed_ys, radii_i, radii_r, orientations):
    r   = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    ang = np.arctan2(ys - cy, xs - cx)
    # half-annulus broken wave → free end → spiral
    mask_u = (r < ri)  & (ang > phi)
    mask_v = (r >= ri) & (r < ro) & (ang > phi)
    u[mask_u] = 1.0
    v[mask_v] = 0.6

print("Warming up simulation…")
for k in range(WARMUP):
    u, v = step(u, v)
    if k % 500 == 0:
        print(f"  {k}/{WARMUP}", end="\r", flush=True)
print()

print("Generating frames…")
frames = []
for i in range(FRAMES):
    for _ in range(STEPS_PER_FRAME):
        u, v = step(u, v)
    frames.append(Image.fromarray(colorize(u, v), "RGB"))
    print(f"  frame {i + 1}/{FRAMES}", end="\r", flush=True)
print()

out = os.path.join(os.path.dirname(__file__), "header-anim.gif")
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=DURATION,
    optimize=False,
)
print(f"Saved → {out}")
