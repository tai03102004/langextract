from paddleocr import PaddleOCR

import numpy as np
import cv2
import re

# Bạn cần biết 2 đường dẫn này từ kết quả fine-tune
REC_MODEL_DIR = "/Users/macbookpro14m1pro/Desktop/RagTable/data/inference_model"   # thư mục chứa model rec

ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    lang="en",
    # ── Load model fine-tune ──────────────────
    # text_recognition_model_dir=REC_MODEL_DIR,
)

def clean_text(text, col_idx=None):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\+\*/=<>()\[\]{}^_%:;!?@#$&/]', '', text)
    return text

def process_single_cell(cell, img_np, H, W, ocr, upscale):
    pad = int(min(cell.w, cell.h) * 0.05)
    x1, y1 = max(0, cell.x - pad), max(0, cell.y - pad)
    x2, y2 = min(W, cell.x + cell.w + pad), min(H, cell.y + cell.h + pad)
    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        cell.text = ""; return cell

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    
    # Phóng to (giữ tỉ lệ)
    cell_area = cell.w * cell.h
    if cell_area < 300: scale = 12
    elif cell_area < 800: scale = 8
    else: scale = upscale
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # CLAHE nhẹ để làm rõ nét mảnh (không làm dày)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Chuẩn hóa về 0-255
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Sharpen vừa phải
    kernel_sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel_sharpen)
    
    img_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = ocr.predict(img_input)
    text = ""
    if result:
        texts = []
        for page in result:
            if isinstance(page, dict) and "rec_texts" in page:
                texts.extend(page["rec_texts"])
            elif isinstance(page, list):
                for line in page:
                    if isinstance(line, list) and len(line) >= 2:
                        t = line[1]
                        if isinstance(t, (list, tuple)):
                            texts.append(t[0])
                        elif isinstance(t, str):
                            texts.append(t)
        text = " ".join(t for t in texts if t)
    cell.text = clean_text(text, cell.col_idx)   
    return cell

def crop_and_ocr_fast(image_pil, cells, ocr, upscale=4, max_workers=8):
    img_np = np.array(image_pil.convert("RGB"))
    H, W = img_np.shape[:2]
    
    for cell in cells:
        process_single_cell(cell, img_np, H, W, ocr, upscale)
    return cells