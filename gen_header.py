"""
Generates an animated GIF header using overlapping wave interference patterns.
Produces a wide banner (800x200) suitable for a GitHub README.
"""
import numpy as np
from PIL import Image
import os

W, H = 800, 200
FRAMES = 48
DURATION = 60  # ms per frame (~16 fps)

x = np.linspace(0, 4 * np.pi, W)
y = np.linspace(0, 2 * np.pi, H)
X, Y = np.meshgrid(x, y)

# Wave source positions (as fractions of the domain)
sources = [
    (0.15, 0.25),
    (0.50, 0.80),
    (0.80, 0.20),
    (0.35, 0.60),
    (0.70, 0.55),
]
SX = [sx * 4 * np.pi for sx, _ in sources]
SY = [sy * 2 * np.pi for _, sy in sources]


def make_field(t):
    val = np.zeros((H, W))
    for cx, cy in zip(SX, SY):
        r = np.hypot(X - cx, Y - cy)
        val += np.sin(r * 2.8 - t)
    val /= len(sources)
    return (val + 1.0) / 2.0  # [0, 1]


def colormap(arr):
    """Blue-cyan-purple palette via HSV interpolation."""
    h = 0.54 + arr * 0.32   # cyan (0.54) → purple (0.86)
    s = 0.85 * np.ones_like(arr)
    v = 0.25 + arr * 0.75

    h6 = h * 6.0
    i = h6.astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t_ = v * (1.0 - s * (1.0 - f))

    r = np.choose(i, [v, q, p, p, t_, v])
    g = np.choose(i, [t_, v, v, q, p, p])
    b = np.choose(i, [p, p, t_, v, v, q])

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


print("Generating frames…")
frames = []
for i in range(FRAMES):
    t = i * 2 * np.pi / FRAMES
    arr = make_field(t)
    rgb = colormap(arr)
    frames.append(Image.fromarray(rgb, mode="RGB"))
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
