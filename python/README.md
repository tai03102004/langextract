# 🧾 RagTable

**Extract tables from images/PDFs to Markdown — fast, clean, RAG-ready**

---

## 🚀 Features

* **🧠 Table structure segmentation** using EfficientUNet (~19M params) with 5 masks: `row`, `col`, `col_header`, `row_header`, `span`
* **🧩 Smart grid completion** — automatically fills missing rows/columns
* **🔍 Cell-level OCR (PaddleOCR)** → reduces noise, improves accuracy
* **📄 Markdown export** → ready for RAG / LLM pipelines
* **🔒 Fully offline** → suitable for sensitive data
* **🖼 Multi-format support:** PNG, JPG, TIFF, PDF

---

## 📦 Installation
```bash
pip install ragtab

Requirements
Python ≥ 3.10

PyTorch (install separately if needed)

PaddleOCR (auto-installed)

⚡ Quickstart

Basic usage (recommended: UNet model)

```python
from ragtab.pipeline import extract_table

markdown, cells = extract_table(
    "bang_mau.png",
    model_path="checkpoints/unet_best.pt",
    ocr_engine="paddleocr"
)

print(markdown)
```

📊 Example Output
```
| STT | Tên sản phẩm | Đơn giá | Số lượng |
| --- | ----------- | ------- | -------- |
| 1   | iPhone 15   | 999     | 12       |
| 2   | Samsung S24 | 899     | 8        |
```

---

## 🧠 How It Works
Input Image (384x384)
       │
       ▼
[1] U-Net Segmentation → row + col masks
       │
       ▼
[2] Projection + filtering → detect rows/columns
       │
       ▼
[3] Grid construction → cell bounding boxes
       │
       ▼
[4] Cell cropping → OCR (PaddleOCR)
       │
       ▼
[5] Markdown output

---

---

## 🔧 Model

RagTable uses a U-Net checkpoint (`.pt`):

* Download the pretrained checkpoint (Google Drive / Hugging Face — contact author) or retrain using the provided notebooks.

* Place the checkpoint in your project directory:

```
checkpoints/
```

Then pass it to the pipeline:

```python
model_path="checkpoints/unet_best.pt"
```

---

## 🗂️ Project Structure

```
RagTable/
├── python/
│   └── ragtab/          # Package chính
│       ├── __init__.py
│       ├── detection.py
│       ├── heuristic.py
│       ├── model.py
│       ├── ocr.py
│       ├── pipeline.py
│       └── utils.py
├── checkpoints/
├── README.md
└── ...
```

---

---

## 📝 License

MIT – free for commercial use.

---

## 👤 Author

**Dinh Duc Tai**
📧 [dinhductai2004@gmail.com](mailto:dinhductai2004@gmail.com)

---
