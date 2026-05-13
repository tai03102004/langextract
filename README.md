<p align="center">
  <img src="data/banner.png" alt="RagTable banner" width="600" />
</p>

<h1 align="center">RagTable</h1>
<p align="center">
  <b>Extract tables from images into Markdown — fast, accurate, RAG-ready</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/ragtab/"><img src="https://img.shields.io/pypi/v/ragtab?color=blue" /></a>
  <a href="https://pypi.org/project/ragtab/"><img src="https://img.shields.io/pypi/pyversions/ragtab" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" /></a>
</p>

---

RagTable converts table images into structured Markdown using a segmentation model + per-cell OCR. It's designed for **borderless and complex tables** (e.g. academic papers, reports) and produces output that plugs directly into RAG pipelines or LLM contexts.

---

## Features

- **5-channel segmentation** via EfficientUNet (~19M params): `row`, `col`, `col_header`, `row_header`, `span`
- **Span detection** — merged cells are detected and rendered with correct `colspan` in Markdown
- **Smart header handling** — header rows use a dedicated OCR pass to improve accuracy on dense or styled text
- **Per-cell OCR** via PaddleOCR — crops each cell individually to minimize cross-cell bleed
- **Post-processing** — removes phantom edge columns, empty rows, and footer artifacts automatically
- **Fully offline** — no external API calls, suitable for sensitive or air-gapped environments
- **Multiple input formats**: PNG, JPG, TIFF

---

## Installation

```bash
pip install ragtab
```

**Requirements:**
- Python ≥ 3.10
- PyTorch (install separately if not already present)
- PaddleOCR (installed automatically)

### Install from source

```bash
git clone https://github.com/tai03102004/rag-table
cd ragtab
pip install -e .
```

---

## Quickstart

```python
from ragtab.pipeline import extract_table

markdown, cells = extract_table(
    "table.png",
    model_path="checkpoints/unet_best.pt",
    ocr_engine="paddleocr"
)

print(markdown)
```

Output:

```
| Item        | Price | Qty |
| ----------- | ----- | --- |
| iPhone 15   | 999   | 12  |
| Samsung S24 | 899   | 8   |
```

---

## How it works

```
Input image (resized to 384×384)
       │
       ▼
[1] EfficientUNet → 5 segmentation masks (row, col, col_header, row_header, span)
       │
       ▼
[2] Projection analysis → row/column separator positions
       │
       ▼
[3] Span detection → connected components on masked region
       │
       ▼
[4] Grid construction → per-cell bounding boxes
       │
       ▼
[5] Header rows → dedicated OCR pass with spatial mapping
[5] Body cells  → per-cell crop + OCR
       │
       ▼
[6] Post-processing → drop phantom columns, empty rows, footers
       │
       ▼
[7] Export → Markdown
```

Each stage is independently accessible so you can swap components or customize the pipeline.

---

## Project structure

```
RagTable/
├── python/
│   └── ragtab/
│       ├── __init__.py
│       ├── detection.py     # Mask → grid cells
│       ├── model.py         # EfficientUNet definition
│       ├── ocr.py           # PaddleOCR wrapper + text cleaning
│       ├── pipeline.py      # End-to-end extract_table()
│       └── utils.py
├── checkpoints/             # Model weights (.pt)
├── notebooks/
│   └── 02_table-recognition.ipynb
└── README.md
```

---

## Model & checkpoints

RagTable uses a U-Net checkpoint (`.pt`). You can:

- **Train from scratch** using `notebooks/02_table-recognition.ipynb`
- **Download a pretrained checkpoint** (Google Drive / Hugging Face)
- **Google Drive: https://drive.google.com/drive/folders/1ILV2zmI6Go-u16bFRbM_Hi1vAyLzc1dQ?usp=drive_link
- **Hugging Face: https://huggingface.co/henryhs/rag-table/blob/main/unet_best.pt

Place the checkpoint at `checkpoints/unet_best.pt` and pass it via `model_path`.

---

## License

MIT — free to use, including for commercial purposes.

---

## Author

**Dinh Duc Tai** — [dinhductai2004@gmail.com](mailto:dinhductai2004@gmail.com)

If you find this project useful, consider giving it a ⭐️ on GitHub!