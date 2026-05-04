import cv2
import numpy as np
from PIL import Image
from .ocr import crop_and_ocr
from .pipeline import cells_to_markdown
from .utils import Cell, ocr


def detect_table_lines(img_cv):
    """Phát hiện đường kẻ ngang và dọc bằng morphological operations."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Kernel ngang dài để bắt đường kẻ ngang
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    # Kernel dọc dài để bắt đường kẻ dọc
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    return h_lines, v_lines


def lines_to_separators(line_mask, axis, min_gap=5):
    """
    Chiếu mask xuống 1 trục, tìm các vị trí có đường kẻ.
    axis=0 → chiếu theo cột (tìm hàng ngang)
    axis=1 → chiếu theo hàng (tìm cột dọc)
    """
    projection = line_mask.sum(axis=axis)
    separators = []
    in_line = False
    start = 0

    for i, val in enumerate(projection):
        if val > 0 and not in_line:
            in_line = True
            start = i
        elif val == 0 and in_line:
            in_line = False
            mid = (start + i) // 2
            if not separators or (mid - separators[-1]) > min_gap:
                separators.append(mid)

    return separators


def bordered_table_extraction(image_path):
    orig_img = Image.open(image_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    h_lines, v_lines = detect_table_lines(img_cv)

    row_seps = lines_to_separators(h_lines, axis=1)
    col_seps = lines_to_separators(v_lines, axis=0)

    if not row_seps or row_seps[0] > 10:
        row_seps = [0] + row_seps
    if row_seps[-1] < orig_h - 10:
        row_seps.append(orig_h)

    if not col_seps or col_seps[0] > 10:
        col_seps = [0] + col_seps
    if col_seps[-1] < orig_w - 10:
        col_seps.append(orig_w)

    cells = []
    for r in range(len(row_seps) - 1):
        for c in range(len(col_seps) - 1):
            y1, y2 = row_seps[r], row_seps[r + 1]
            x1, x2 = col_seps[c], col_seps[c + 1]

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            cells.append(Cell(
                row_idx=r, col_idx=c,
                x=x1, y=y1, w=x2 - x1, h=y2 - y1
            ))

    cells = crop_and_ocr(orig_img, cells, ocr)
    return cells_to_markdown(cells)