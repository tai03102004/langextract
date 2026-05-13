import re
from typing import List
from .utils import Cell


def drop_empty_edge_columns(cells: List[Cell]) -> List[Cell]:
    """Loại bỏ cột rỗng hoàn toàn ở rìa trái/phải (phantom từ col_mask)."""
    if not cells:
        return cells
    max_col = max(c.col_idx + c.col_span for c in cells)

    # Cột nào có text không-rỗng?
    has_content = [False] * max_col
    for cell in cells:
        if cell.text.strip():
            for c in range(cell.col_idx, cell.col_idx + cell.col_span):
                if c < max_col:
                    has_content[c] = True

    # Tìm cột rỗng liên tục ở rìa trái và rìa phải
    left_drop = 0
    while left_drop < max_col and not has_content[left_drop]:
        left_drop += 1
    right_drop = max_col
    while right_drop > left_drop and not has_content[right_drop - 1]:
        right_drop -= 1

    if left_drop == 0 and right_drop == max_col:
        return cells  # không có phantom

    out = []
    for cell in cells:
        old_start = cell.col_idx
        old_end = cell.col_idx + cell.col_span - 1
        # Clip vào range [left_drop, right_drop - 1]
        new_start = max(old_start, left_drop)
        new_end = min(old_end, right_drop - 1)
        if new_start > new_end:
            continue  # cell hoàn toàn nằm trong vùng bị drop
        cell.col_idx = new_start - left_drop
        cell.col_span = new_end - new_start + 1
        cell.is_span = cell.col_span > 1
        out.append(cell)
    # print(f"  Dropped {left_drop} cột trái + {max_col - right_drop} cột phải")
    return out

def drop_empty_rows(cells: List[Cell]) -> List[Cell]:
    """Xoá row hoàn toàn rỗng và renumber row_idx."""
    from collections import defaultdict
    row_has_text = defaultdict(bool)
    for c in cells:
        if c.text.strip():
            row_has_text[c.row_idx] = True

    keep = [c for c in cells if row_has_text[c.row_idx]]
    if not keep:
        return []

    # Renumber row_idx liên tục từ 0
    old_rows = sorted({c.row_idx for c in keep})
    remap = {old: new for new, old in enumerate(old_rows)}
    for c in keep:
        c.row_idx = remap[c.row_idx]
    return keep


def drop_footer_rows(cells: List[Cell]) -> List[Cell]:
    """Xoá row chứa pattern footer (Author manuscript, PMC, doi, ...)."""
    from collections import defaultdict
    row_texts = defaultdict(list)
    for c in cells:
        row_texts[c.row_idx].append(c.text)

    FOOTER_RE = re.compile(
        r'author\s*manuscript|available\s*in\s*pmc|^\s*doi[:\s]|chem\s*phys',
        re.IGNORECASE
    )
    bad = {r for r, ts in row_texts.items() if FOOTER_RE.search(" ".join(ts))}
    return [c for c in cells if c.row_idx not in bad]