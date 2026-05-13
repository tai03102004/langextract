from typing import List
import cv2
from .utils import Cell
import numpy as np

def detect_header_cells_via_ocr(image_pil, row_sep, col_sep, header_row_indices,
                                  ocr, orig_w, orig_h, img_size):
    img_np = np.array(image_pil.convert("RGB"))
    sx = orig_w / img_size
    sy = orig_h / img_size
    col_sep_orig = [int(c * sx) for c in col_sep]
    n_cols = len(col_sep_orig) - 1

    # Ngưỡng gap để mở rộng box (tỉ lệ theo orig_w để generalize qua ảnh kích thước khác)
    GAP_THRESHOLD = max(40, int(orig_w * 0.12))
    LEFT_BOUND = col_sep_orig[0]
    RIGHT_BOUND = col_sep_orig[-1]

    cells = []
    for r in header_row_indices:
        y1 = int(row_sep[r] * sy)
        y2 = int(row_sep[r + 1] * sy)
        if y2 - y1 < 8:
            continue

        strip = img_np[y1:y2, :]
        scaled = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        results = ocr.predict([scaled])
        if not results or not isinstance(results[0], dict):
            continue
        res = results[0]
        rec_texts = res.get("rec_texts", [])
        boxes = res.get("rec_polys") or res.get("rec_boxes") or res.get("dt_polys") or []

        # ── Step 1: collect text boxes với x-range ─────
        raw_boxes = []
        for text, box in zip(rec_texts, boxes):
            text = text.strip()
            if not text:
                continue
            box = np.asarray(box, dtype=float)
            if box.ndim == 1 and box.size == 4:
                xl, xr = box[0], box[2]
            else:
                xl, xr = box[:, 0].min(), box[:, 0].max()
            raw_boxes.append((xl / 3, xr / 3, text))
        raw_boxes.sort(key=lambda b: b[0])

        # ── Step 1.5: mở rộng x-range về phía neighbor nếu có gap lớn ──
        expanded_boxes = []
        for i, (xl, xr, text) in enumerate(raw_boxes):
            left_neighbor = raw_boxes[i - 1][1] if i > 0 else LEFT_BOUND
            if xl - left_neighbor > GAP_THRESHOLD:
                new_xl = (left_neighbor + xl) / 2
            else:
                new_xl = xl
            right_neighbor = raw_boxes[i + 1][0] if i + 1 < len(raw_boxes) else RIGHT_BOUND
            if right_neighbor - xr > GAP_THRESHOLD:
                new_xr = (xr + right_neighbor) / 2
            else:
                new_xr = xr
            expanded_boxes.append((new_xl, new_xr, text))

        # ── Step 2: map mỗi box (đã mở rộng) → (col_start, col_end) ─
        # Dùng overlap / cell_width thay vì overlap / box_width
        mapped = []
        for xl, xr, text in expanded_boxes:
            xc = (xl + xr) / 2
            col_start, col_end = None, None
            for c in range(n_cols):
                cl, cr = col_sep_orig[c], col_sep_orig[c + 1]
                cell_width = max(1, cr - cl)
                overlap = max(0, min(xr, cr) - max(xl, cl))
                if overlap / cell_width > 0.5 or (cl <= xc < cr):
                    if col_start is None:
                        col_start = c
                    col_end = c
            if col_start is None:
                continue
            mapped.append((col_start, col_end, text, xc))

        # ── Step 3: gộp boxes có col_range CHỒNG NHAU ──
        mapped.sort(key=lambda m: (m[0], m[3]))
        merged_boxes = []
        for cs, ce, txt, xc in mapped:
            if merged_boxes:
                pcs, pce, ptxt, pxc = merged_boxes[-1]
                if cs <= pce and ce >= pcs:
                    merged_boxes[-1] = (
                        min(pcs, cs), max(pce, ce),
                        ptxt + " " + txt, (pxc + xc) / 2
                    )
                    continue
            merged_boxes.append((cs, ce, txt, xc))

        # ── Step 4: tạo Cell, fill cell rỗng cho col trống ──
        covered = [False] * n_cols
        row_cells = []
        for cs, ce, text, _ in merged_boxes:
            for c in range(cs, ce + 1):
                covered[c] = True
            row_cells.append(Cell(
                row_idx=r, col_idx=cs,
                x=col_sep_orig[cs], y=y1,
                w=col_sep_orig[ce + 1] - col_sep_orig[cs], h=y2 - y1,
                is_span=(ce > cs), col_span=ce - cs + 1,
                text=text,
            ))

        c = 0
        while c < n_cols:
            if covered[c]:
                c += 1
                continue
            c_end = c
            while c_end + 1 < n_cols and not covered[c_end + 1]:
                c_end += 1
            for cc in range(c, c_end + 1):
                row_cells.append(Cell(
                    row_idx=r, col_idx=cc,
                    x=col_sep_orig[cc], y=y1,
                    w=col_sep_orig[cc + 1] - col_sep_orig[cc], h=y2 - y1,
                    is_span=False, col_span=1, text="",
                ))
            c = c_end + 1

        cells.extend(row_cells)

    return cells
def masks_to_cell_boxes(row_mask, col_mask, span_mask,
                        col_header_mask=None, row_header_mask=None,
                        orig_w=384, orig_h=384, img_size=384) -> List[Cell]:
    # ── 1. Tách line → row_sep, col_sep ───
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (img_size // 4, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_size // 4))
    k3  = np.ones((3, 3), np.uint8)

    h_lines = cv2.dilate(cv2.morphologyEx(row_mask * 255, cv2.MORPH_OPEN, k_h), k3, iterations=1)
    v_lines = cv2.dilate(cv2.morphologyEx(col_mask * 255, cv2.MORPH_OPEN, k_v), k3, iterations=1)

    def mask_to_positions(proj, min_gap=6):
        norm = proj / (proj.max() + 1e-6)
        is_line = norm > 0.03
        lines, in_line, start = [], False, 0
        for i, v in enumerate(is_line):
            if v and not in_line:
                in_line, start = True, i
            elif not v and in_line:
                in_line = False
                mid = (start + i) // 2
                if not lines or mid - lines[-1] >= min_gap:
                    lines.append(mid)
        return lines

    row_sep = [0] + mask_to_positions(h_lines.sum(axis=1).astype(float)) + [img_size]
    col_sep = [0] + mask_to_positions(v_lines.sum(axis=0).astype(float)) + [img_size]

    num_rows = len(row_sep) - 1
    num_cols = len(col_sep) - 1

    # ── 2. Cắt span_mask tại các grid separator ────────
    boundary = np.zeros_like(span_mask, dtype=np.uint8)
    wall = 2  # nửa độ rộng tường (pixel)
    for x in col_sep[1:-1]:
        boundary[:, max(0, x - wall):x + wall + 1] = 1
    for y in row_sep[1:-1]:
        boundary[max(0, y - wall):y + wall + 1, :] = 1

    span_binary = (span_mask > 0.5).astype(np.uint8)
    span_cut = span_binary * (1 - boundary)

    # ── 3. Connected components trên mask đã cắt ───────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(span_cut, connectivity=8)

    # Lọc CC nhỏ
    valid_labels = set()
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            valid_labels.add(i)

    print(f"  → {num_rows} rows, {num_cols} cols, "
          f"{len(valid_labels)} span region(s) sau khi cắt")

    # ── 4. Ánh xạ mỗi grid cell tới 1 component ────────
    cell_comp = np.zeros((num_rows, num_cols), dtype=np.int32)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, y2 = row_sep[r], row_sep[r + 1]
            x1, x2 = col_sep[c], col_sep[c + 1]
            region = labels[y1:y2, x1:x2]
            if region.size == 0:
                continue
            # Lấy label trội (bỏ qua 0 = background)
            nonzero = region[region > 0]
            if nonzero.size == 0:
                continue
            uniq, cnts = np.unique(nonzero, return_counts=True)
            dom = int(uniq[cnts.argmax()])
            if dom in valid_labels and cnts.max() > 0.3 * region.size:
                cell_comp[r, c] = dom

    # ── 5. Build cells: merge các ô liên tiếp cùng comp_id ─
    sx, sy = orig_w / img_size, orig_h / img_size
    visited = np.zeros((num_rows, num_cols), dtype=bool)
    cells = []

    for r in range(num_rows):
        c = 0
        while c < num_cols:
            if visited[r, c]:
                c += 1
                continue

            comp = cell_comp[r, c]
            c_end = c
            if comp > 0:
                # Merge sang phải nếu CÙNG component
                while c_end + 1 < num_cols and cell_comp[r, c_end + 1] == comp:
                    c_end += 1
            # comp == 0 → cell bình thường, không merge

            for cc in range(c, c_end + 1):
                visited[r, cc] = True

            y1, y2 = row_sep[r], row_sep[r + 1]
            x1, x2 = col_sep[c], col_sep[c_end + 1]
            col_span = c_end - c + 1

            if (x2 - x1) >= 6 and (y2 - y1) >= 6:
                cells.append(Cell(
                    row_idx=r, col_idx=c,
                    x=int(x1 * sx), y=int(y1 * sy),
                    w=int((x2 - x1) * sx), h=int((y2 - y1) * sy),
                    is_span=(col_span > 1),
                    col_span=col_span,
                ))
            c = c_end + 1

    return cells, row_sep, col_sep