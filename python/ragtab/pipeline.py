from PIL import Image
from typing import List

from .utils import Cell, ocr
import numpy as np
import torch
import matplotlib.pyplot as plt

from .ocr import crop_and_ocr
from .detection import masks_to_cell_boxes
from .model import EfficientUNet
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
def image_to_markdown_v3(image_path, model, device, img_size=384, upscale=3):
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    print(f"Ảnh gốc: {orig_w}×{orig_h}px")

    # Resize trực tiếp (không padding)
    draft = TF.resize(orig_img, (img_size, img_size),
                      interpolation=TF.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(draft).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(img_t)).squeeze(0).cpu().numpy()

    # Ngưỡng cố định 0.5 cho tất cả mask
    row_mask  = (preds[0] > 0.5).astype(np.uint8)
    col_mask  = (preds[1] > 0.5).astype(np.uint8)
    span_mask = (preds[4] > 0.5).astype(np.uint8)

    # Gọi hàm masks_to_cell_boxes MỚI
    cells = masks_to_cell_boxes(row_mask, col_mask, span_mask,
                                orig_w, orig_h, img_size)
    print(f"Tổng cells: {len(cells)}")

    # OCR
    cells = crop_and_ocr(orig_img, cells, ocr, upscale=upscale)

    # Xuất Markdown
    md = cells_to_markdown(cells)
    return md, cells
    

def extract_table(image_path, model_path, ocr_engine="paddleocr"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientUNet(out_ch=5, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
        
    return image_to_markdown_v3(image_path, model, device, img_size=384, upscale=3)
