from typing import List
import cv2
from matplotlib.table import Cell
import numpy as np

def straighten_row_mask(row_mask, min_width_ratio=0.25, 
                        line_thickness=1, close_kernel_size=(1,5)):
    kernel = np.ones(close_kernel_size, np.uint8)
    row_mask_closed = cv2.morphologyEx(row_mask, cv2.MORPH_CLOSE, kernel)
    H, W = row_mask_closed.shape
    proj = row_mask_closed.sum(axis=1).astype(float)
    threshold = W * min_width_ratio
    active = (proj > threshold).astype(np.uint8)
    straightened = np.zeros_like(row_mask)
    
    separators = []  
    in_region, start = False, 0
    for y in range(H):
        if active[y] and not in_region:
            in_region, start = True, y
        elif not active[y] and in_region:
            in_region = False
            center = (start + y) // 2
            separators.append(center)
            straightened[center, :] = 1  
    if in_region:
        center = (start + H) // 2
        separators.append(center)
        straightened[center, :] = 1
    
    return straightened, separators  

def straighten_col_mask(col_mask, min_height_ratio=0.25, line_thickness=1):
    H, W = col_mask.shape
    proj = col_mask.sum(axis=0).astype(float)
    threshold = H * min_height_ratio
    active = (proj > threshold).astype(np.uint8)
    straightened = np.zeros_like(col_mask)
    
    separators = []
    in_region, start = False, 0
    for x in range(W):
        if active[x] and not in_region:
            in_region, start = True, x
        elif not active[x] and in_region:
            in_region = False
            center = (start + x) // 2
            separators.append(center)
            straightened[:, center] = 1
    if in_region:
        center = (start + W) // 2
        separators.append(center)
        straightened[:, center] = 1
    
    return straightened, separators
def detect_rows_from_whitespace(pil_img, orig_w, orig_h, img_size=384,
                                effective_w=None, effective_h=None):
    """
    Trả về danh sách vị trí Y (trong không gian ảnh gốc) của các đường phân cách hàng
    dựa vào projection dọc của ảnh xám.
    """
    if effective_w is None: effective_w = img_size
    if effective_h is None: effective_h = img_size

    gray = np.array(pil_img.convert('L'))           
    v_proj = gray.sum(axis=1).astype(np.float64)
    norm = v_proj / (v_proj.max() + 1e-6)
    is_blank = norm > 0.85

    min_gap = max(2, int(orig_h * 0.005))  
    lines = []
    in_gap, start = False, 0
    for i, v in enumerate(is_blank):
        if v and not in_gap:
            in_gap, start = True, i
        elif not v and in_gap:
            in_gap = False
            mid = (start + i) // 2
            if not lines or mid - lines[-1] >= min_gap:
                lines.append(mid)
    if in_gap:
        mid = (start + len(is_blank) - 1) // 2
        if not lines or mid - lines[-1] >= min_gap:
            lines.append(mid)
    return lines   

def detect_cols_from_whitespace(orig_img_pil, row_sep, orig_w, orig_h,
                                 img_size, effective_w, effective_h):
    """
    khi col_mask yếu → tìm cột bằng vertical whitespace
    """
    img_np = np.array(orig_img_pil.convert("L")) 
    H, W   = img_np.shape

    _, bw = cv2.threshold(img_np, 200, 255, cv2.THRESH_BINARY)

    vert_proj = (bw < 200).sum(axis=0).astype(float)

    vert_proj = np.convolve(vert_proj,
                            np.ones(5)/5, mode='same')

    threshold = vert_proj.max() * 0.05
    is_space  = vert_proj < threshold

    col_dividers = []
    in_space, start = False, 0
    for i, v in enumerate(is_space):
        if v and not in_space:
            in_space, start = True, i
        elif not v and in_space:
            in_space = False
            width = i - start
            if width > 8:  
                col_dividers.append((start + i) // 2)

    return [0] + col_dividers + [W]

def masks_to_cell_boxes_v2(row_mask, col_mask,
                           orig_w, orig_h, img_size=384,
                           effective_w=None, effective_h=None,
                           orig_img_pil=None) -> List[Cell]:
    if effective_w is None: effective_w = img_size
    if effective_h is None: effective_h = img_size

    row_mask = row_mask[:effective_h, :effective_w]
    col_mask = col_mask[:effective_h, :effective_w]

    # ── Row projection ───────────────────────────────────
    row_straight, _ = straighten_row_mask(row_mask, min_width_ratio=0.25)
    h_proj = row_straight.sum(axis=1).astype(float)

    # ── Col projection ───────────────────────────────────
    v_proj = col_mask.astype(float).sum(axis=0)
    v_proj = np.convolve(v_proj, np.ones(3)/3, mode='same')

    def mask_to_positions(proj, effective_dim, min_gap_ratio=0.02, threshold_ratio=0.01):
        min_gap = max(3, int(effective_dim * min_gap_ratio))
        norm = proj / (proj.max() + 1e-6)
        is_line = norm > threshold_ratio
        lines, in_line, start = [], False, 0
        for i, v in enumerate(is_line):
            if v and not in_line:
                in_line, start = True, i
            elif not v and in_line:
                in_line = False
                mid = (start + i) // 2
                if not lines or mid - lines[-1] >= min_gap:
                    lines.append(mid)
        if in_line:
            mid = (start + len(is_line) - 1) // 2
            if not lines or mid - lines[-1] >= min_gap:
                lines.append(mid)
        return lines

    row_lines = mask_to_positions(h_proj, effective_dim=effective_h,
                                  min_gap_ratio=0.01, threshold_ratio=0.02)
    col_lines = mask_to_positions(v_proj, effective_dim=effective_w,
                                  min_gap_ratio=0.01, threshold_ratio=0.02)

    # ── BƯỚC 1: tính row_sep ─────────────────────────────
    min_dist_to_bottom = max(3, int(effective_h * 0.03))
    row_lines = [l for l in row_lines if (effective_h - l) > min_dist_to_bottom]
    row_sep = [0] + row_lines + [effective_h]

    # ── BƯỚC 2: bổ sung whitespace rows nếu thiếu ────────
    if orig_img_pil is not None:
        MIN_EXPECTED_ROWS = 5
        avg_h = effective_h / max(1, len(row_sep)-1)
        last_h = row_sep[-1] - row_sep[-2]
        if len(row_sep)-1 < MIN_EXPECTED_ROWS or last_h > 2.5 * avg_h:
            white_lines = detect_rows_from_whitespace(
                orig_img_pil, orig_w, orig_h,
                img_size, effective_w, effective_h
            )
            white_lines_eff = [int(y * effective_h / orig_h) for y in white_lines]
            if white_lines_eff:
                extra = [e for e in white_lines_eff if row_sep[-2] < e < effective_h]
                if extra:
                    row_sep = row_sep[:-1] + extra + [effective_h]
                    row_sep = sorted(set(row_sep))
            print(f"  → Bổ sung whitespace rows, tổng {len(row_sep)-1} hàng")

    # ── BƯỚC 3: bổ sung whitespace cols (SAU row_sep) ────
    if orig_img_pil is not None:
        ws_cols = detect_cols_from_whitespace(
            orig_img_pil, row_sep, orig_w, orig_h,
            img_size, effective_w, effective_h
        )
        ws_cols_eff = [int(x * effective_w / orig_w) for x in ws_cols
                       if 0 < x < orig_w]
        MIN_DIST = max(5, int(effective_w * 0.03))
        added = 0
        for ws in ws_cols_eff:
            if all(abs(ws - cl) > MIN_DIST for cl in col_lines):
                col_lines.append(ws)
                added += 1
        if added:
            col_lines.sort()
            print(f"  → Bổ sung {added} cột từ whitespace, tổng {len(col_lines)+1} cols")

    # ── BƯỚC 4: tính col_sep ─────────────────────────────
    col_mask_strength = col_mask.sum() / (effective_w * effective_h)
    if col_mask_strength < 0.01 or len(col_lines) < 1:
        print("  ⚠️ col_mask yếu, dùng whitespace detection")
        col_sep_orig = detect_cols_from_whitespace(
            orig_img_pil, row_sep, orig_w, orig_h,
            img_size, effective_w, effective_h
        )
        col_sep = [int(x / (orig_w/effective_w)) for x in col_sep_orig]
        col_sep = [0] + [c for c in col_sep if 0 < c < effective_w] + [effective_w]
    else:
        col_sep = [0] + col_lines + [effective_w]

    print(f"  → {len(row_sep)-1} rows, {len(col_sep)-1} cols")

    sx = orig_w / effective_w
    sy = orig_h / effective_h
    cells = []
    for r in range(len(row_sep) - 1):
        for c in range(len(col_sep) - 1):
            x1, x2 = col_sep[c], col_sep[c+1]
            y1, y2 = row_sep[r], row_sep[r+1]
            cells.append(Cell(
                row_idx=r, col_idx=c,
                x=int(x1*sx), y=int(y1*sy),
                w=int((x2-x1)*sx), h=int((y2-y1)*sy)
            ))

    print("\n=== DEBUG CELLS ROW 1 ===")
    for c in cells:
        if c.row_idx == 1:
            print(f"  col {c.col_idx}: span={c.row_span}x{c.col_span}, x={c.x}, y={c.y}, w={c.w}, h={c.h}")

    return cells