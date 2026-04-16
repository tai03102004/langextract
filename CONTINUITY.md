# CONTINUITY.md — Table Extraction & U-Net Segmentation Project

## Goal
- So sánh 6 phương pháp extract table từ PDF/image cho RAG pipeline
- Train U-Net (EfficientNet-B4 encoder) để predict 5 mask channels: `row`, `col`, `col_header`, `row_header`, `span`
- Pipeline hoàn chỉnh: Segmentation → Grid → OCR → JSON/Markdown

**Success criteria:**
- mIoU ≥ 0.80 trên validation set
- So sánh vs baseline: Tesseract, Rule-based, Table Transformer

## Data
- Local dataset: `/Users/macbookpro14m1pro/Desktop/RagTable/python/data/pubtabnet/` (200 PNG + JSON samples)
- Kaggle dataset: `tiinh123/table-segmentation-data` (30,000 images + 5-channel masks)
  - Masks: 5 loại → `row`, `col`, `col_header`, `row_header`, `span`
  - col mask đã dilated 1 iteration để fix thin-line issue
  - Backup: `tiinh123/table-seg-masks-dilated` (149,826 files)

## Architecture
- **Model:** EfficientUNet (EfficientNet-B4 encoder + U-Net decoder)
- **Input:** 384×384 RGB image
- **Output:** 5-channel binary mask (H×W×5)
- **Params:** 19.2M | **Batch:** 16 (T4) / 8 (macOS M-Series)

## Training State
- Kaggle: epoch=45, best_mIoU=0.7674 (trên 30,000 samples)
- macOS M1: train được, cần downscale batch_size=2-4
- Checkpoint: `checkpoints_effb4_ft/best_model.pt`

## Pipeline Steps (Bước 2→5)
1. **Bước 2 - Structure Segmentation:** U-Net predict 5 masks
2. **Bước 3 - Post-processing:** masks → grid cells (row/column intersection)
3. **Bước 4 - OCR:** đọc text từng cell bằng EasyOCR
4. **Bước 5 - Reconstruction:** ghép structure + text → JSON/Markdown

## State
- **Done:** Dataset analysis (30k masks), Model definition (EfficientUNet), Kaggle training (epoch=45, mIoU=0.7674)
- **Now:** Viết lại notebook clean (local paths), hoàn thiện Bước 2-5 pipeline
- **Next:** Chạy inference local → đánh giá mIoU → so sánh vs 6 methods

## Open Questions
- mIoU 0.7674 vs target 0.80 → cần thêm epochs hay tăng col weight?
- Post-processing: intersection grid algorithm hay dùng row/col mask directly?
- macOS M1 Pro: inference speed benchmark cần thiết?

## Working Set
- Project root: `/Users/macbookpro14m1pro/Desktop/RagTable/`
- Python code: `python/`
- Notebooks: `python/notebooks/01_explore_and_compare.ipynb`, `02_table-recognition.ipynb`
- Models: Kaggle `tiinh123/table-segmentation-data` | Local `python/checkpoints/`
- Venv: `python/.venv/`
