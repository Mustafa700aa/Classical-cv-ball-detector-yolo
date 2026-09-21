"""
=============================================================================
Task 1.2: Detect the Pattern - Red & Blue Ball Detector
=============================================================================
Author : MIA Phase 5
Approach: Classical Computer Vision using OpenCV (no deep learning)

Pipeline:
  1. Preprocessing  - CLAHE on LAB L-channel + Gaussian blur
  2. Color Masking  - HSV thresholding (dual-range red, single-range blue)
  3. Morphology     - Open (remove noise) then Close (fill gaps/shadows)
  4. Contour Filter - Circularity, relative-area, aspect-ratio shape tests
  5. IoU Dedup      - Remove overlapping duplicate detections
  6. Label Output   - YOLO-format normalized .txt files
  7. Annotate       - Draw bounding boxes on a copy of each image
=============================================================================
"""

import cv2
import numpy as np
import os
import glob
import shutil

# ---------------------------------------------------------------------------
# CONFIGURATION  (tune here if needed without touching logic)
# ---------------------------------------------------------------------------

# HSV color ranges - (H_lo, H_hi, S_lo, S_hi, V_lo, V_hi)
RED_RANGE_1  = (  0,  12,  80, 255,  50, 255)   # Lower red hue band
RED_RANGE_2  = (158, 180,  80, 255,  50, 255)   # Upper red hue band
BLUE_RANGE   = ( 95, 130,  80, 255,  40, 255)   # Blue hue band

# Morphological kernel sizes
MORPH_OPEN_K  = 5    # Ellipse kernel for opening  (remove noise specks)
MORPH_CLOSE_K = 15   # Ellipse kernel for closing  (fill shadow holes)

# Contour shape filters
# MIN_AREA_RATIO: ball contour must be at least this fraction of image area.
# This scales automatically with image resolution.
# A ball ~5% of image width -> circle area ~ pi*(0.05*W)^2 ~ 0.008 * W^2
# We use 0.0015 as conservative lower bound to catch small/far balls.
MIN_AREA_RATIO  = 0.0001    # relative to image area (enables detection of small/far balls in high-res images)
MAX_AREA_RATIO  = 0.70      # relative to image area (filters huge blobs)
MIN_CIRCULARITY = 0.40      # Circularity gate (4*pi*A / P^2)
MIN_FILL_RATIO  = 0.40      # contour_area / enclosing_circle_area
MAX_ASPECT_DIFF = 0.45      # abs(w/h - 1) must be below this

# IoU threshold for duplicate suppression
IOU_THRESHOLD = 0.30

# Paths - dynamically resolve whether run from repo root or parent folder
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR     = os.path.join(_SCRIPT_DIR, "balls") if os.path.isdir(os.path.join(_SCRIPT_DIR, "balls")) else "balls"
OUTPUT_LABELS = os.path.join(_SCRIPT_DIR, "output", "labels")
OUTPUT_ANNOT  = os.path.join(_SCRIPT_DIR, "output", "annotated")

# Class definitions
CLASS_BLUE = 0
CLASS_RED  = 1
CLASS_COLORS = {
    CLASS_BLUE: (220,  80,  20),   # BGR: dark-orange box for blue balls
    CLASS_RED:  ( 20,  20, 220),   # BGR: red box for red balls
}
CLASS_NAMES = {CLASS_BLUE: "Blue", CLASS_RED: "Red"}


# ---------------------------------------------------------------------------
# STEP 1: PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess(img):
    """
    Normalize brightness with CLAHE on the L-channel of LAB color space,
    then apply Gaussian blur to suppress pixel-level noise.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) locally
    equalizes the lightness channel so balls in shadow or harsh sunlight
    reach a more uniform brightness without distorting hue/saturation.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_blur  = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    return img_blur


# ---------------------------------------------------------------------------
# STEP 2: COLOR MASKING
# ---------------------------------------------------------------------------

def build_masks(img_preprocessed):
    """
    Convert to HSV and threshold for red (two ranges) and blue (one range).

    Red wraps around BOTH ends of the hue wheel (0-12 and 158-180 degrees)
    so two inRange calls are OR-combined into one red mask.
    Blue sits in the 95-130 degree band and needs only one call.
    """
    hsv = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2HSV)

    lower_r1 = np.array([RED_RANGE_1[0], RED_RANGE_1[2], RED_RANGE_1[4]], dtype=np.uint8)
    upper_r1 = np.array([RED_RANGE_1[1], RED_RANGE_1[3], RED_RANGE_1[5]], dtype=np.uint8)
    lower_r2 = np.array([RED_RANGE_2[0], RED_RANGE_2[2], RED_RANGE_2[4]], dtype=np.uint8)
    upper_r2 = np.array([RED_RANGE_2[1], RED_RANGE_2[3], RED_RANGE_2[5]], dtype=np.uint8)
    lower_b  = np.array([BLUE_RANGE[0],  BLUE_RANGE[2],  BLUE_RANGE[4]],  dtype=np.uint8)
    upper_b  = np.array([BLUE_RANGE[1],  BLUE_RANGE[3],  BLUE_RANGE[5]],  dtype=np.uint8)

    red_mask  = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1),
                               cv2.inRange(hsv, lower_r2, upper_r2))
    blue_mask = cv2.inRange(hsv, lower_b, upper_b)
    return red_mask, blue_mask


# ---------------------------------------------------------------------------
# STEP 3: MORPHOLOGICAL CLEANUP
# ---------------------------------------------------------------------------

def clean_mask(mask):
    """
    Opening  (erode then dilate) removes small isolated noise pixels.
    Closing  (dilate then erode) fills holes caused by dark panel lines,
    shadows, or specular highlights on the ball surface.
    Elliptical kernels are used to match the round ball shape.
    """
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_OPEN_K,  MORPH_OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_CLOSE_K, MORPH_CLOSE_K))
    opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed


# ---------------------------------------------------------------------------
# STEP 4: CONTOUR DETECTION & SHAPE FILTERING
# ---------------------------------------------------------------------------

def is_ball_shaped(cnt, img_area):
    """
    Four shape tests to decide if a contour represents a ball:

    1. Relative area gate  - scales with image resolution; rejects tiny
                             noise blobs and massive background regions.
    2. Circularity         - 4*pi*A / P^2, close to 1 for circles.
    3. Fill ratio          - contour area vs its minimum-enclosing-circle.
    4. Aspect ratio        - bounding rect width/height must be near 1.0.
    """
    area = cv2.contourArea(cnt)

    # 1. Relative area gate
    if area < MIN_AREA_RATIO * img_area:
        return False
    if area > MAX_AREA_RATIO * img_area:
        return False

    # 2. Circularity
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < MIN_CIRCULARITY:
        return False

    # 3. Fill ratio vs minimum enclosing circle
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * radius ** 2
    if circle_area > 0 and (area / circle_area) < MIN_FILL_RATIO:
        return False

    # 4. Aspect ratio of bounding rect
    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0 or abs(w / h - 1.0) > MAX_ASPECT_DIFF:
        return False

    return True


def detect_from_mask(mask, class_id, img_h, img_w):
    """
    Find contours in cleaned binary mask, filter for ball shape, and
    return list of YOLO-format normalized detections:
      [(class_id, x_center, y_center, width, height), ...]
    All coordinates are normalized to [0, 1].
    """
    img_area = img_h * img_w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if not is_ball_shaped(cnt, img_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Clamp bounding box to image boundaries
        x = max(0, x);  y = max(0, y)
        w = min(w, img_w - x);  h = min(h, img_h - y)
        xc = (x + w / 2) / img_w
        yc = (y + h / 2) / img_h
        detections.append((class_id, xc, yc, w / img_w, h / img_h))
    return detections


# ---------------------------------------------------------------------------
# STEP 5: IoU-BASED DUPLICATE SUPPRESSION
# ---------------------------------------------------------------------------

def _iou(boxA, boxB):
    """Compute Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _to_abs(det, img_w, img_h):
    """Convert normalized (xc, yc, w, h) to absolute (x1, y1, x2, y2)."""
    _, xc, yc, w, h = det
    return ((xc - w/2)*img_w, (yc - h/2)*img_h,
            (xc + w/2)*img_w, (yc + h/2)*img_h)


def remove_duplicates(detections, img_w, img_h):
    """
    Greedy Non-Maximum Suppression:
    Keep the largest detection; suppress anything that overlaps it by
    more than IOU_THRESHOLD. Also removes blue detections that are
    fully contained within a larger red detection (logo patches).
    """
    if not detections:
        return []

    # Sort by box area descending
    detections = sorted(detections, key=lambda d: d[3] * d[4], reverse=True)
    kept = []
    for det in detections:
        box = _to_abs(det, img_w, img_h)
        suppress = False
        for kdet in kept:
            kbox = _to_abs(kdet, img_w, img_h)
            if _iou(box, kbox) > IOU_THRESHOLD:
                suppress = True
                break
            # Suppress if this box is almost entirely inside a kept box
            inter_x1 = max(box[0], kbox[0]);  inter_y1 = max(box[1], kbox[1])
            inter_x2 = min(box[2], kbox[2]);  inter_y2 = min(box[3], kbox[3])
            inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
            this_area  = (box[2]-box[0]) * (box[3]-box[1])
            if this_area > 0 and inter_area / this_area > 0.75:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# STEP 6: LABEL FILE GENERATION
# ---------------------------------------------------------------------------

def save_labels(detections, out_path):
    """
    Write a YOLO-format label file. Each line:
      <class_id> <x_center> <y_center> <width> <height>
    All spatial values are normalized to [0, 1].
    """
    with open(out_path, 'w') as f:
        for det in detections:
            class_id, xc, yc, w, h = det
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# STEP 7: ANNOTATED IMAGE
# ---------------------------------------------------------------------------

def annotate_image(img, detections, img_w, img_h):
    """
    Draw colored bounding boxes and class name labels on a copy of the image.
    Blue-labeled boxes for blue balls, red-labeled boxes for red balls.
    """
    out = img.copy()
    for det in detections:
        class_id, xc, yc, w, h = det
        x1 = int((xc - w/2) * img_w);  y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w);  y2 = int((yc + h/2) * img_h)
        color = CLASS_COLORS[class_id]
        label = CLASS_NAMES[class_id]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def process_image(img_path, label_dir, annot_dir):
    """
    Run the full detection pipeline on a single image.
    Returns the number of balls detected.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Cannot read: {img_path}")
        return 0

    img_h, img_w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]

    # Step 1: Preprocess
    processed = preprocess(img)

    # Step 2: Color masks
    red_mask, blue_mask = build_masks(processed)

    # Step 3: Morphological cleanup
    red_clean  = clean_mask(red_mask)
    blue_clean = clean_mask(blue_mask)

    # Step 4: Detect balls from each mask
    red_dets  = detect_from_mask(red_clean,  CLASS_RED,  img_h, img_w)
    blue_dets = detect_from_mask(blue_clean, CLASS_BLUE, img_h, img_w)

    # Step 5: Remove duplicates / overlapping boxes
    all_dets = remove_duplicates(red_dets + blue_dets, img_w, img_h)

    # Step 6: Save label file
    save_labels(all_dets, os.path.join(label_dir, base + ".txt"))

    # Step 7: Save annotated image
    annotated = annotate_image(img, all_dets, img_w, img_h)
    cv2.imwrite(os.path.join(annot_dir, os.path.basename(img_path)), annotated)

    r_count = sum(1 for d in all_dets if d[0] == CLASS_RED)
    b_count = sum(1 for d in all_dets if d[0] == CLASS_BLUE)
    print(f"  {os.path.basename(img_path):22s}  {len(all_dets):2d} ball(s)"
          f"  [Red:{r_count}  Blue:{b_count}]")
    return len(all_dets)


def main():
    print("=" * 62)
    print("   Task 1.2  -  Red & Blue Ball Detector (Classical CV)")
    print("=" * 62)

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(
            f"Input directory not found: '{INPUT_DIR}'\n"
            "Please place images in Task5_1.2/balls/")

    for d in [OUTPUT_LABELS, OUTPUT_ANNOT]:
        os.makedirs(d, exist_ok=True)

    # Gather all image files, sorted numerically
    patterns  = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(INPUT_DIR, p)))

    def sort_key(p):
        name = os.path.splitext(os.path.basename(p))[0]
        for part in reversed(name.split("_")):
            if part.isdigit():
                return int(part)
        return name

    img_paths = sorted(img_paths, key=sort_key)

    if not img_paths:
        raise FileNotFoundError(f"No images found in '{INPUT_DIR}'")

    print(f"\nProcessing {len(img_paths)} image(s) from '{INPUT_DIR}' ...\n")

    total = sum(process_image(p, OUTPUT_LABELS, OUTPUT_ANNOT)
                for p in img_paths)

    print(f"\n{'='*62}")
    print(f"   Done! {total} total ball(s) detected across all images.")
    print(f"   Label files  -> {OUTPUT_LABELS}/")
    print(f"   Annotated    -> {OUTPUT_ANNOT}/")
    print("=" * 62)

    # Create submission ZIP containing only the label files
    zip_name = os.path.join(_SCRIPT_DIR, "submission_labels")
    shutil.make_archive(zip_name, "zip", OUTPUT_LABELS)
    print(f"\n   Submission ZIP -> {zip_name}.zip")


if __name__ == "__main__":
    main()
