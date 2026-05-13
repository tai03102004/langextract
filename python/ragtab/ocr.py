import numpy as np
import cv2
import re 
import math

import numpy as np
import cv2
import re 
import math

def clean_text(text, col_idx=None):
    text = text.strip()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\-\(\)%±/=+:;|&'<>~]", "", text)
    return text

def fix_percent_96(text):
    return re.sub(r"(\d+\.\d)96\b", r"\1%", text)


def fix_percent_tail(text):
    return re.sub(r"%\d+", "%", text)


def fix_missing_percent(text):
    def repl(match):
        val = match.group(1)
        if "%" not in val and re.match(r"\d+\.\d+", val):
            return f"({val}%)"
        return match.group()
    return re.sub(r"\((.*?)\)", repl, text)

def fix_pm_spacing(text):
    return re.sub(r"(\d)±\s*(\d)", r"\1 ± \2", text)

def normalize_percent(text):
    def repl(match):
        val = float(match.group(1))

        if val < 0.1:
            return match.group()

        return f"{val:.1f}%"
    return re.sub(r"(\d+\.\d+)%", repl, text)

def truncate(val, decimals=2):
    factor = 10 ** decimals
    return math.floor(val * factor) / factor

def normalize_small_percent(text):
    def repl(match):
        val = float(match.group(1))
        if val < 0.1:
            val = truncate(val, 2)
            return f"{val:.2f}%"
        return match.group()
    return re.sub(r"(\d+\.\d+)%", repl, text)

def fix_numbers(text):
    text = fix_percent_tail(text)
    text = fix_percent_96(text)
    # text = fix_missing_percent(text)
    
    text = normalize_small_percent(text)
    text = normalize_percent(text)

    text = fix_pm_spacing(text)

    return text


def crop_and_ocr(image_pil, cells, ocr, upscale=3):
    img_np = np.array(image_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    batch_imgs = []
    valid_cells = []

    for cell in cells:
        if cell.text:          
            continue
        pad = max(int(min(cell.w, cell.h) * 0.15), 3)
        x1 = max(0, cell.x - pad); y1 = max(0, cell.y - pad)
        x2 = min(W, cell.x + cell.w + pad); y2 = min(H, cell.y + cell.h + pad)

        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            cell.text = ""
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        min_scale = 3.0
        max_scale = 5.0
        h_ratio = max(0.3, min(1.0, cell.h / 100.0)) 
        scale = min_scale + (1 - h_ratio) * (max_scale - min_scale)

        new_w = int(gray.shape[1] * scale)
        new_h = int(gray.shape[0] * scale)

        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        img_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_input = resize_with_padding(img_input, target_height=64, target_width=256)

        batch_imgs.append(img_input)
        valid_cells.append(cell)

    results = ocr.predict(batch_imgs)

    for cell, result in zip(valid_cells, results):
        texts = []
        if result:
            if isinstance(result, dict) and "rec_texts" in result:
                texts.extend(result["rec_texts"])
        text = " ".join(t for t in texts if t)

        text = clean_text(text, cell.col_idx)
        text = fix_numbers(text)

        cell.text = text
    return cells

def resize_with_padding(img, target_height, target_width):
    """Resize ảnh giữ nguyên tỉ lệ, thêm viền trắng nếu cần để đạt kích thước mục tiêu."""
    h, w = img.shape[:2]
    ratio = min(target_width / w, target_height / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Tạo canvas trắng
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    # Tính offset để căn giữa
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas