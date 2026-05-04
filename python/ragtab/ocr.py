import numpy as np
import cv2
import re 
import math

def clean_text(text, col_idx=None):
    text = text.strip()

    text = text.replace(",", ".")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\-\(\)%±/=+:;|&]", "", text)
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
        pad = int(min(cell.w, cell.h) * 0.15)
        x1 = max(0, cell.x - pad); y1 = max(0, cell.y - pad)
        x2 = min(W, cell.x + cell.w + pad); y2 = min(H, cell.y + cell.h + pad)

        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            cell.text = ""
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        scale = 4 if cell.row_idx == 0 else upscale  
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        img_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        TARGET_H, TARGET_W = 64, 256
        img_input = cv2.resize(img_input, (TARGET_W, TARGET_H))

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