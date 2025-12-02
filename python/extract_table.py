import csv
import os
import io
import pandas as pd
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import Image

# Đường dẫn tesseract (thay đổi theo máy bạn nếu cần)
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract/tessdata/'

# Đường dẫn đến file PDF
src = "../data/test.pdf"

# 1️ Chuyển PDF sang ảnh (mỗi trang là 1 ảnh)
pages = convert_from_path(src, dpi=400)  

# 2️ Khởi tạo OCR
ocr = TesseractOCR(n_threads=2, lang="eng")
docs = []

# 3️ Lặp qua từng trang và trích xuất bảng
for i, page in enumerate(pages, start=1):
    # Chuyển trang (PIL.Image) thành byte buffer
    img_bytes = io.BytesIO()
    page.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Tạo đối tượng Image từ img2table
    doc = Image(img_bytes)

    # Trích xuất bảng
    extracted_tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=30
    )

    def clean_text(text):
        """Làm sạch ký tự xuống dòng và thay dấu phẩy trong nội dung"""
        if text is None:
            return ""
        text = str(text).replace("\n", "")
        text = text.replace(",", "·")
        return text.strip()

    for idx, table in enumerate(extracted_tables, start=1):
        print(f"\n--- Table {idx} ---")
        df = table.df.copy()
        idx += 1
        # Header
        header_row = df.iloc[0].apply(clean_text).tolist()
        docs.append(header_row)

        # Dữ liệu
        for row in df.iloc[1:].itertuples(index=False):
            cleaned_row = [clean_text(cell) for cell in row]
            docs.append(cleaned_row)


df_final = pd.DataFrame(docs)
df_final.to_csv(
    "rotated_table_2.csv",
    index=False,
    header=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL
)

print("✅ Saved rotated_table_2.csv (all tables combined)")
