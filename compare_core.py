import cv2
import numpy as np


def resize_to_match(img1: np.ndarray, img2: np.ndarray):
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1, img2


def compute_change_mask(img1: np.ndarray, img2: np.ndarray, diff_threshold: int = 45):
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


def draw_changes(img: np.ndarray, mask: np.ndarray, min_area: int = 1200):
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


def compare_images(img1_bgr: np.ndarray, img2_bgr: np.ndarray,
                   diff_threshold: int = 45, min_area: int = 1200):
    img1, img2 = resize_to_match(img1_bgr, img2_bgr)
    mask = compute_change_mask(img1, img2, diff_threshold)
    annotated, regions = draw_changes(img2, mask, min_area)
    return mask, annotated, regions
