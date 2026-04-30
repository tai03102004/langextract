from tkinter import Image
from typing import List

from matplotlib.table import Cell
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from python.ragtab import ocr
from python.ragtab.detection import masks_to_cell_boxes_v2
from python.ragtab.ocr import crop_and_ocr_fast
import torch
import torchvision.transforms.functional as TF

def cells_to_markdown(cells: List[Cell]) -> str:
    if not cells:
        return ""
    max_row = max(c.row_idx for c in cells) + 1
    max_col = max(c.col_idx for c in cells) + 1
    grid = [[""] * max_col for _ in range(max_row)]
    for cell in sorted(cells, key=lambda c: (c.row_idx, c.col_idx)):
        grid[cell.row_idx][cell.col_idx] = cell.text
    lines = []
    for r, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if r == 0:
            lines.append("| " + " | ".join(["---"] * max_col) + " |")
    return "\n".join(lines)
# ── Pipeline hoàn chỉnh ───────────────────────────────────
def image_to_markdown_v3(image_path, model1, model2, device,
                          thresholds, img_size=384, upscale=2.5):
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    scale  = img_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w*scale), int(orig_h*scale)
    draft  = orig_img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (img_size, img_size), (255,255,255))
    padded.paste(draft, (0,0))

    img_t = TF.to_tensor(padded).unsqueeze(0).to(device)
    model1.eval(); model2.eval()
    with torch.no_grad():
        p1 = torch.sigmoid(model1(img_t))
        p2 = torch.sigmoid(model2(img_t))
        preds = ((p1 + p2) / 2).squeeze(0).cpu().numpy()

    row_mask = (preds[0] > thresholds["row"]).astype(np.uint8)
    col_mask = (preds[1] > thresholds["col"]).astype(np.uint8)

    cells = masks_to_cell_boxes_v2(
        row_mask, col_mask,
        orig_w, orig_h, img_size,
        effective_w=new_w, effective_h=new_h,
        orig_img_pil=orig_img   
    )
    print(f"  Cells: {len(cells)}")

    cells = crop_and_ocr_fast(orig_img, cells, ocr, upscale=upscale)
    print("\n=== ALL CELLS AFTER OCR ===")
    for c in sorted(cells, key=lambda x: (x.row_idx, x.col_idx)):
        print(f"row {c.row_idx}, col {c.col_idx}: text='{c.text}'")
    return cells_to_markdown(cells), cells
