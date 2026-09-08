"""
synthetic_data.py -- Generates synthetic X-ray-like images for demo/testing.
Mirrors the real SIXray folder structure so every other module works unchanged.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def _xray_background(h: int, w: int) -> np.ndarray:
    """Baggage background: gradient + Perlin-like noise."""
    base = np.random.normal(loc=180, scale=12, size=(h, w)).clip(0, 255)
    # soft gradient to simulate X-ray lamp falloff
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    vignette = 1 - 0.4 * (dist / dist.max())
    return (base * vignette).clip(0, 255).astype(np.uint8)


def _draw_ellipse(arr: np.ndarray, cx: int, cy: int,
                  rx: int, ry: int, intensity: int):
    h, w = arr.shape
    Y, X = np.ogrid[:h, :w]
    mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
    arr[mask] = np.clip(arr[mask].astype(int) - intensity, 0, 255)


def _draw_rectangle(arr, x0, y0, x1, y1, intensity):
    arr[y0:y1, x0:x1] = np.clip(
        arr[y0:y1, x0:x1].astype(int) - intensity, 0, 255)


def _draw_line(arr, x0, y0, length, thickness, angle_deg, intensity):
    """Simulate a knife/scissors blade as a thin dark stripe."""
    import cv2
    angle_rad = np.deg2rad(angle_deg)
    x1 = int(x0 + length * np.cos(angle_rad))
    y1 = int(y0 + length * np.sin(angle_rad))
    overlay = arr.copy()
    cv2.line(overlay, (x0, y0), (x1, y1), int(intensity), thickness)
    alpha = 0.8
    arr[:] = np.clip(
        alpha * overlay + (1 - alpha) * arr, 0, 255).astype(np.uint8)


def make_prohibited_image(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic prohibited-item image: bag + one dark object silhouette."""
    img = _xray_background(h, w)
    # bag boundary
    _draw_ellipse(img, w // 2, h // 2, w // 2 - 10, h // 2 - 10, 30)
    # random item type
    item = np.random.choice(["knife", "gun", "scissors"])
    rng = np.random
    if item == "knife":
        import cv2
        x0, y0 = rng.randint(40, w - 80), rng.randint(40, h - 40)
        length = rng.randint(50, 90)
        angle = rng.randint(0, 180)
        overlay = img.copy()
        x1 = int(x0 + length * np.cos(np.deg2rad(angle)))
        y1 = int(y0 + length * np.sin(np.deg2rad(angle)))
        cv2.line(overlay, (x0, y0), (x1, y1), 30, 3)
        img = (0.8 * overlay + 0.2 * img).astype(np.uint8)
    elif item == "gun":
        import cv2
        x0, y0 = rng.randint(40, w - 70), rng.randint(40, h - 50)
        cv2.rectangle(img, (x0, y0), (x0 + 55, y0 + 20), 35, -1)
        cv2.rectangle(img, (x0 + 10, y0 + 20), (x0 + 25, y0 + 40), 40, -1)
    else:  # scissors
        import cv2
        cx, cy = rng.randint(50, w - 50), rng.randint(50, h - 50)
        for angle in [30, -30]:
            x1 = int(cx + 40 * np.cos(np.deg2rad(angle)))
            y1 = int(cy + 40 * np.sin(np.deg2rad(angle)))
            cv2.line(img, (cx, cy), (x1, y1), 35, 3)
    img = img + np.random.normal(0, 5, img.shape).astype(np.int32)
    return img.clip(0, 255).astype(np.uint8)


def make_safe_image(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic safe-bag image: bag + random benign objects."""
    img = _xray_background(h, w)
    _draw_ellipse(img, w // 2, h // 2, w // 2 - 10, h // 2 - 10, 20)
    n_items = np.random.randint(1, 5)
    for _ in range(n_items):
        cx = np.random.randint(40, w - 40)
        cy = np.random.randint(40, h - 40)
        rx = np.random.randint(10, 30)
        ry = np.random.randint(8, 20)
        _draw_ellipse(img, cx, cy, rx, ry, np.random.randint(15, 35))
    img = img + np.random.normal(0, 5, img.shape).astype(np.int32)
    return img.clip(0, 255).astype(np.uint8)


def generate_synthetic_dataset(root: str,
                                n_positive: int = 800,
                                n_negative: int = 4000,
                                image_size: int = 224):
    """
    Generates synthetic X-ray images under:
        root/positive/  -- images with prohibited items
        root/negative/  -- safe bag images
    """
    pos_dir = Path(root) / "positive"
    neg_dir = Path(root) / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already generated
    existing_pos = len(list(pos_dir.glob("*.png")))
    existing_neg = len(list(neg_dir.glob("*.png")))
    if existing_pos >= n_positive and existing_neg >= n_negative:
        print(f"[SyntheticData] Already exists: {existing_pos} pos, "
              f"{existing_neg} neg -- skipping generation.")
        return

    print(f"[SyntheticData] Generating {n_positive} positive images...")
    for i in tqdm(range(n_positive), desc="Positive"):
        arr = make_prohibited_image(image_size, image_size)
        # Convert grayscale -> RGB (3-channel) for consistency
        rgb = np.stack([arr, arr, arr], axis=-1)
        Image.fromarray(rgb).save(pos_dir / f"pos_{i:05d}.png")

    print(f"[SyntheticData] Generating {n_negative} negative images...")
    for i in tqdm(range(n_negative), desc="Negative"):
        arr = make_safe_image(image_size, image_size)
        rgb = np.stack([arr, arr, arr], axis=-1)
        Image.fromarray(rgb).save(neg_dir / f"neg_{i:05d}.png")

    print(f"[SyntheticData] Done. Folder: {root}")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    generate_synthetic_dataset(root)
