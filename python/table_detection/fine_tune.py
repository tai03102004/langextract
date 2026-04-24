import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

# ===== CẤU HÌNH =====
DATA_DIR = Path("../../data/PaddleOCR")          # thư mục chứa images/ và annotations/
CROPS_DIR = Path("../../data/PaddleOCR/pubtabnet_crops")   # nơi lưu ảnh crop
LABEL_DIR = Path("../../data/PaddleOCR/pubtabnet_labels")  # nơi lưu file .txt
CROPS_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

def tokens_to_text(tokens):
    """Chuyển tokens thành text thuần (bỏ HTML tags)"""
    return ''.join(t for t in tokens if not t.startswith('<') and not t.startswith('</')).strip()

def process_one_sample(img_path, anno_path, output_crop_dir, label_writer):
    """Xử lý một cặp ảnh + annotation, crop các ô và ghi label"""
    # Đọc annotation
    with open(anno_path, encoding='utf-8') as f:
        anno = json.load(f)
    
    # Lấy thông tin cells (lưu ý: trường 'html' là string chứa dict)
    html_data = eval(anno['html'])   # chuyển string thành dict (cẩn thận: eval chỉ dùng khi tin tưởng dữ liệu)
    cells = html_data.get('cells', [])
    
    # Mở ảnh gốc
    img = Image.open(img_path).convert('RGB')
    
    sample_id = img_path.stem   # ví dụ 'sample_000001'
    for idx, cell in enumerate(cells):
        bbox = cell.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # Thêm padding nhẹ (2px)
        pad = 2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.width, x2 + pad)
        y2 = min(img.height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        
        crop = img.crop((x1, y1, x2, y2))
        crop_filename = f"{sample_id}_cell_{idx:04d}.png"
        crop_path = output_crop_dir / crop_filename
        crop.save(crop_path)
        
        # Lấy text ground truth
        text = tokens_to_text(cell.get('tokens', []))
        if text:
            # Ghi vào file label (sẽ append sau)
            label_writer.write(f"{crop_path.relative_to(LABEL_DIR.parent)}\t{text}\n")

def main():
    # Lấy danh sách tất cả ảnh và annotation
    img_dir = DATA_DIR / "images"
    anno_dir = DATA_DIR / "annotations"
    img_paths = sorted(img_dir.glob("*.png"))
    
    # Random shuffle và chia train/val (90/10)
    random.seed(42)
    random.shuffle(img_paths)
    split_idx = int(len(img_paths) * 0.9)
    train_imgs = img_paths[:split_idx]
    val_imgs = img_paths[split_idx:]
    
    # Mở file label cho train và val
    train_label_file = LABEL_DIR / "train_list.txt"
    val_label_file = LABEL_DIR / "val_list.txt"
    
    with open(train_label_file, 'w', encoding='utf-8') as f_train, \
         open(val_label_file, 'w', encoding='utf-8') as f_val:
        
        # Xử lý train
        print(f"Đang xử lý {len(train_imgs)} ảnh train...")
        for img_path in tqdm(train_imgs):
            anno_path = anno_dir / f"{img_path.stem}.json"
            if not anno_path.exists():
                continue
            # Tạo thư mục crop riêng cho train (để dễ quản lý)
            crop_dir = CROPS_DIR / "train"
            crop_dir.mkdir(parents=True, exist_ok=True)
            process_one_sample(img_path, anno_path, crop_dir, f_train)
        
        # Xử lý val
        print(f"Đang xử lý {len(val_imgs)} ảnh val...")
        for img_path in tqdm(val_imgs):
            anno_path = anno_dir / f"{img_path.stem}.json"
            if not anno_path.exists():
                continue
            crop_dir = CROPS_DIR / "val"
            crop_dir.mkdir(parents=True, exist_ok=True)
            process_one_sample(img_path, anno_path, crop_dir, f_val)
    
    print(f"✅ Hoàn tất! Train labels: {train_label_file}, Val labels: {val_label_file}")
    print(f"📁 Ảnh crop được lưu trong {CROPS_DIR}")

if __name__ == "__main__":
    main()