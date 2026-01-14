import cv2
import numpy as np


def resize_to_match(img1: np.ndarray, img2: np.ndarray):
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1, img2


def compute_change_mask(
    img1: np.ndarray,
    img2: np.ndarray,
    diff_threshold: int = 60,        # ↑ less sensitive (was 45)
    blur_ksize: int = 7,             # ↑ stronger blur (was (5,5))
    open_iter: int = 2,              # ↑ stronger noise removal (was 1)
    close_iter: int = 3              # ↑ connect regions better (was 2)
):
    """
    Less-sensitive version:
    - stronger blur -> reduces pixel-level noise
    - higher threshold -> keeps only clearer differences
    - stronger morphology cleanup
    """
    a = cv2.GaussianBlur(img1, (blur_ksize, blur_ksize), 0)
    b = cv2.GaussianBlur(img2, (blur_ksize, blur_ksize), 0)

    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)

    return mask


def draw_changes(img: np.ndarray, mask: np.ndarray, min_area: int = 1500):  # ↑ less sensitive (was 1200)
    annotated = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    return annotated, regions


def compare_images(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    diff_threshold: int = 60,   # ↑ default less sensitive
    min_area: int = 1500        # ↑ default less sensitive
):
    img1, img2 = resize_to_match(img1_bgr, img2_bgr)
    mask = compute_change_mask(img1, img2, diff_threshold=diff_threshold)
    annotated, regions = draw_changes(img2, mask, min_area=min_area)
    return mask, annotated, regions
