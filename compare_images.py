import os
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MVP resize: make img2 same size as img1."""
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1, img2


def compute_change_mask(img1: np.ndarray, img2: np.ndarray, diff_threshold: int = 45) -> np.ndarray:
    """
    Robust for colorful images / screenshots:
    blur -> absdiff -> grayscale -> threshold -> morphology cleanup.
    Returns binary mask (255 = change).
    """
    a = cv2.GaussianBlur(img1, (5, 5), 0)
    b = cv2.GaussianBlur(img2, (5, 5), 0)

    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)

    return mask


def draw_changes(img: np.ndarray, mask: np.ndarray, min_area: int = 1200) -> tuple[np.ndarray, list[dict]]:
    """
    Draw red boxes around significant changed regions.
    Returns annotated image + regions list.
    """
    annotated = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[dict] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

        regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    return annotated, regions


def main():
    # ---- paths (change here) ----
    img1_path = "images/map1.png"
    img2_path = "images/map3.png"  # or map2.png

    # ---- debug (helps if VS Code runs wrong file / wrong folder) ----
    print("RUNNING FILE:", __file__)
    print("CWD:", os.getcwd())
    print("img1 exists?", Path(img1_path).exists(), "-", img1_path)
    print("img2 exists?", Path(img2_path).exists(), "-", img2_path)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1, img2 = resize_to_match(img1, img2)

    # Tune these two numbers if needed
    mask = compute_change_mask(img1, img2, diff_threshold=45)
    annotated, regions = draw_changes(img2, mask, min_area=1200)

    cv2.imwrite(str(out_dir / "change_mask.png"), mask)
    cv2.imwrite(str(out_dir / "map_with_boxes.png"), annotated)

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"Detected regions: {len(regions)}")
    for r in regions:
        print(r)


if __name__ == "__main__":
    main()
