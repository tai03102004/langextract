from typing import List
import cv2
from .utils import Cell
import numpy as np

def masks_to_cell_boxes(row_mask, col_mask, span_mask,
                        orig_w, orig_h, img_size=384) -> List[Cell]:
    # Kernel cực lớn để chỉ giữ lại các đường dài (loại bỏ chữ)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (img_size//4, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_size//4))
    k3  = np.ones((3, 3), np.uint8)

    # Tách đường ngang / dọc
    h_lines = cv2.morphologyEx(row_mask * 255, cv2.MORPH_OPEN, k_h)
    v_lines = cv2.morphologyEx(col_mask * 255, cv2.MORPH_OPEN, k_v)

    # Nối đứt nhẹ
    h_lines = cv2.dilate(h_lines, k3, iterations=1)
    v_lines = cv2.dilate(v_lines, k3, iterations=1)

    def mask_to_positions(proj, min_gap=6):
        norm    = proj / (proj.max() + 1e-6)
        is_line = norm > 0.03
        lines   = []
        in_line, start = False, 0
        for i, v in enumerate(is_line):
            if v and not in_line:
                in_line, start = True, i
            elif not v and in_line:
                in_line = False
                mid = (start + i) // 2
                if not lines or mid - lines[-1] >= min_gap:
                    lines.append(mid)
        return lines

    h_proj  = h_lines.sum(axis=1).astype(float)
    v_proj  = v_lines.sum(axis=0).astype(float)

    row_sep = [0] + mask_to_positions(h_proj) + [img_size]
    col_sep = [0] + mask_to_positions(v_proj) + [img_size]

    print(f"  → {len(row_sep)-1} rows, {len(col_sep)-1} cols")

    sx, sy = orig_w / img_size, orig_h / img_size

    cells = []
    for r in range(len(row_sep) - 1):
        for c in range(len(col_sep) - 1):
            y1, y2 = row_sep[r], row_sep[r+1]
            x1, x2 = col_sep[c], col_sep[c+1]
            if (x2-x1) < 6 or (y2-y1) < 6:
                continue
            rx = int(x1 * sx); ry = int(y1 * sy)
            rw = int((x2-x1) * sx); rh = int((y2-y1) * sy)

            # Kiểm tra span nếu vùng này có span_mask
            span_region = span_mask[y1:y2, x1:x2]
            is_span = span_region.mean() > 0.5 if span_region.size > 0 else False

            cells.append(Cell(
                row_idx=r, col_idx=c,
                x=rx, y=ry, w=rw, h=rh,
                is_span=is_span
            ))
    return cells