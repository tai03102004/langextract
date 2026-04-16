# RagTable — Table Extraction for RAG Pipeline

Extract structured tables from PDFs/images using the best method for your RAG workflow.

## Architecture Overview

```
Input PDF/Image
      │
      ▼
┌─────────────────────────────────────────────┐
│  Step 1: Compare 6 Baseline Methods         │
│  (Docling / img2table / Tesseract /         │
│   Rule-based / Table Transformer / UNet)    │
└──────────────────┬──────────────────────────┘
                   │ best method selected
                   ▼
┌─────────────────────────────────────────────┐
│  Step 2: U-Net Structure Segmentation       │
│  5-channel mask: row / col / col_header /   │
│  row_header / span                          │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Step 3: Grid Reconstruction                │
│  Row/col masks → cell bounding boxes        │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Step 4: OCR per Cell                       │
│  EasyOCR / Tesseract — isolate text         │
└──────────────────┬──────────────────────────┘
                   ▼
            JSON / Markdown
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| **Step 1** | Compare 6 table extraction methods on PubTabNet samples → F1 / speed / accuracy |
| **Step 2** | U-Net (EfficientNet-B4) predicts 5 semantic masks |
| **Step 3** | Post-process masks → extract grid cell bounding boxes |
| **Step 4** | OCR text per cell → avoid cross-cell noise |
| **Step 5** | Reconstruct structured JSON or Markdown table |

## Quick Start

```bash
cd /Users/macbookpro14m1pro/Desktop/RagTable/python
source .venv/bin/activate

# Step 1: Compare all baseline methods
jupyter notebook notebooks/01_explore_and_compare.ipynb

# Step 2-5: U-Net training + inference pipeline
jupyter notebook notebooks/02_table-recognition.ipynb
```

## Datasets

| Dataset | Source | Size | Content |
|---------|--------|------|---------|
| `pubtabnet/` | Local project | 200 samples | PNG + JSON (bbox + HTML tokens) |
| `tiinh123/table-segmentation-data` | Kaggle | 30,000 | Images + 5-channel masks |

## Model

- **Architecture:** EfficientUNet (EfficientNet-B4 encoder + U-Net decoder)
- **Input:** 384×384 RGB
- **Output:** 5 binary masks — `row`, `col`, `col_header`, `row_header`, `span`
- **Parameters:** 19.2M
- **Best mIoU:** 0.7674 (epoch 45, Kaggle T4)

## Package Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `docling` | 2.84.0 | Table extraction from PDF |
| `img2table` | 1.4.2 | Image-based table extraction |
| `easyocr` | 1.7.2 | Text recognition |
| `pytesseract` | 0.3.13 | OCR (requires Tesseract 5.x) |
| `torch` | 2.11.0 | Training & inference |
| `transformers` | 5.4.0 | Table Transformer model |
| `opencv-python` | 4.13.0 | Image processing |
| `timm` | latest | EfficientNet encoder |
| `easyocr` | latest | Per-cell OCR |
| `scipy` | latest | Morphological ops (mask dilation) |

## Project Structure

```
RagTable/
├── README.md
├── CONTINUITY.md
├── python/
│   ├── notebooks/
│   │   ├── 01_explore_and_compare.ipynb   # Step 1: 6-method comparison
│   │   └── 02_table-recognition.ipynb     # Steps 2-5: U-Net pipeline
│   ├── data/pubtabnet/                     # Local test set
│   ├── checkpoints/                        # Saved model weights
│   ├── .venv/                              # Python virtual env
│   └── rag/                                # RAG integration code
└── server.js                               # API server (Node.js)
```

## Comparison Metrics (Step 1)

Each method is evaluated on:
- **Precision / Recall / F1** — cell-level IoU ≥ 0.5 vs ground truth
- **Speed** — seconds per image
- **Qualitative** — side-by-side visual overlay

## Inference (Steps 2-5)

```python
from pathlib import Path
from inference_pipeline import TablePipeline

pipeline = TablePipeline(
    model_path="checkpoints/best_model.pt",
    ocr_engine="easyocr"   # or "tesseract"
)

result = pipeline.process("path/to/table_image.png")
# result = {"cells": [...], "html": "...", "markdown": "..."}
```

## Environment Setup (macOS)

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install docling img2table easyocr pytesseract \
  torch torchvision timm opencv-python scipy transformers

# Install Tesseract OCR (macOS)
brew install tesseract
```

## Environment Setup (Kaggle)

Kernel: **GPU P100** | Runtime: ~40 epochs in ~2h (T4, batch=16)

Dataset inputs required:
- `tiinh123/table-segmentation-data` — images + masks
- `tiinh123/table-seg-masks-dilated` — dilated col masks
- `tiinh123/table-seg-checkpoints-v2` — saved checkpoints

## Next Steps

- [ ] Complete Bước 3: Grid reconstruction algorithm (row/col intersection)
- [ ] Complete Bước 4: Per-cell OCR pipeline
- [ ] Complete Bước 5: JSON / Markdown output
- [ ] Benchmark mIoU on local 200-sample dataset
- [ ] Integrate into RAG pipeline via `server.js`
