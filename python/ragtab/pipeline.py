from PIL import Image
from typing import List

from .utils import Cell, ocr
import numpy as np
import torch

from .ocr import crop_and_ocr
from .detection import detect_header_cells_via_ocr, masks_to_cell_boxes
from .model import EfficientUNet
import torchvision.transforms.functional as TF
from .drop import drop_empty_edge_columns, drop_empty_rows, drop_footer_rows
def cells_to_markdown(cells):
    if not cells:
        return ""
    max_row = max(c.row_idx for c in cells) + 1
    max_col = max(c.col_idx + c.col_span for c in cells)
    grid = [[None] * max_col for _ in range(max_row)]
    SPAN_MARKER = "__SPAN__"

    for cell in sorted(cells, key=lambda c: (c.row_idx, c.col_idx)):
        grid[cell.row_idx][cell.col_idx] = cell.text
        for off in range(1, cell.col_span):
            if cell.col_idx + off < max_col:
                grid[cell.row_idx][cell.col_idx + off] = SPAN_MARKER

    lines = []
    for r, row in enumerate(grid):
        cells_str = [("" if v in (None, SPAN_MARKER) else v) for v in row]
        lines.append("| " + " | ".join(cells_str) + " |")
        if r == 0:
            lines.append("| " + " | ".join(["---"] * max_col) + " |")
    return "\n".join(lines)
# ── Pipeline hoàn chỉnh ───────────────────────────────────
def image_to_markdown_v3(image_path, model, device, img_size=384, upscale=3):
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    # print(f"Ảnh gốc: {orig_w}×{orig_h}px")

    draft = TF.resize(orig_img, (img_size, img_size),
                      interpolation=TF.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(draft).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(img_t)).squeeze(0).cpu().numpy()

    # Debug masks
    # mask_names = ['row', 'col', 'col_header', 'row_header', 'span']
    # fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    # for i, ax in enumerate(axes):
    #     ax.imshow(preds[i], cmap='hot', interpolation='nearest')
    #     ax.set_title(mask_names[i]); ax.axis('off')
    # plt.tight_layout(); plt.savefig('debug_masks.png'); plt.show()

    row_mask        = (preds[0] > 0.5).astype(np.uint8)
    col_mask        = (preds[1] > 0.5).astype(np.uint8)
    col_header_mask = (preds[2] > 0.5).astype(np.uint8)
    row_header_mask = (preds[3] > 0.5).astype(np.uint8)
    span_mask       = (preds[4] > 0.5).astype(np.uint8)

    # ── Cells từ mask + row_sep/col_sep ────────────────
    cells, row_sep, col_sep = masks_to_cell_boxes(
        row_mask, col_mask, span_mask,
        col_header_mask, row_header_mask,
        orig_w, orig_h, img_size
    )

    # ── Xác định header rows từ col_header_mask + row_sep THẬT ──
    num_rows_total = len(row_sep) - 1
    header_row_indices = []
    for r in range(num_rows_total):
        y1, y2 = row_sep[r], row_sep[r + 1]
        region = col_header_mask[y1:y2]
        if region.size > 0 and region.mean() > 0.5:
            header_row_indices.append(r)
    # print(f"Header rows: {header_row_indices}")

    # ── Override header cells bằng OCR-driven detection ────
    if header_row_indices:
        header_cells = detect_header_cells_via_ocr(
            orig_img, row_sep, col_sep, header_row_indices,
            ocr, orig_w, orig_h, img_size=img_size
        )
        cells = [c for c in cells if c.row_idx not in header_row_indices]
        cells.extend(header_cells)

    # ── OCR cho phần còn lại (header cells đã có text, sẽ skip) ──
    cells = crop_and_ocr(orig_img, cells, ocr, upscale=upscale)
    cells = drop_empty_edge_columns(cells) 
    cells = drop_footer_rows(cells)         
    cells = drop_empty_rows(cells)

    # Sort lại để cells_to_markdown đặt đúng thứ tự
    cells.sort(key=lambda c: (c.row_idx, c.col_idx))

    md = cells_to_markdown(cells)
    return md, cells
    

def extract_table(image_path, model_path, ocr_engine="paddleocr"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientUNet(out_ch=5, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"✅ Loaded checkpoint")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
        
    return image_to_markdown_v3(image_path, model, device, img_size=384, upscale=3)
