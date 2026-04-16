import csv
import os
import io
import time
import pandas as pd
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import Image

# Đường dẫn tesseract (thay đổi theo máy bạn nếu cần)
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract/tessdata/'

def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    col_counts = {}
    
    for col in df.columns:
        col_str = str(col).strip()
        
        # Add suffix if duplicate
        if col_str in col_counts:
            col_counts[col_str] += 1
            new_col = f"{col_str}_{col_counts[col_str]}"
        else:
            col_counts[col_str] = 0
            new_col = col_str
        
        cols.append(new_col)
    
    df.columns = cols
    return df


def extract_tables_ocr(pdf_path: str, output_csv: str = None) -> dict:

    start_time = time.time()
    # 1️ Chuyển PDF sang ảnh (mỗi trang là 1 ảnh)
    pages = convert_from_path(pdf_path, dpi=400)  

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

        if not extracted_tables:
            print(f"   ⚠️  Page {i}: No bordered tables, trying borderless...")
            extracted_tables = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=True,
                min_confidence=20 
            )

        def clean_text(text):
            """Làm sạch ký tự xuống dòng"""
            if text is None:
                return ""
            text = str(text).replace("\n", "")
            return text.strip()

        for idx, table in enumerate(extracted_tables, start=1):
            df = table.df.copy()
            if df.empty:
                continue

            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            
            # Apply cleaning
            df = df.applymap(clean_text)
            df = make_columns_unique(df)
            
            docs.append(df)
            print(f"   📊 Page {i}, Table {idx}: {df.shape}")

    processing_time = time.time() - start_time

    if output_csv and docs:
        combined_df = pd.concat(docs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"   💾 Saved to: {output_csv}")
    
    return {
        'method': 'OCR (img2table)',
        'time': processing_time,
        'tables_found': len(docs),
        'status': 'success',
        'dataframes': docs
    }
def extract_tables_from_image(image_path: str, output_csv: str = None) -> dict:
    start_time = time.time()
    
    # Initialize OCR
    ocr = TesseractOCR(n_threads=2, lang="eng")
    docs = []

    # Process the image directly
    doc = Image(image_path)

    # Extract tables
    extracted_tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=30
    )

    if not extracted_tables:
        print(f"   ⚠️  No bordered tables found, trying borderless...")
        extracted_tables = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=20 
        )

    def clean_text(text):
        """Làm sạch ký tự xuống dòng"""
        if text is None:
            return ""
        text = str(text).replace("\n", "")
        return text.strip()

    for idx, table in enumerate(extracted_tables, start=1):
        df = table.df.copy()
        if df.empty:
            continue

        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Apply cleaning
        df = df.applymap(clean_text)
        df = make_columns_unique(df)
        
        docs.append(df)
        print(f"   📊 Table {idx}: {df.shape}")

    processing_time = time.time() - start_time

    if output_csv and docs:
        combined_df = pd.concat(docs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"   💾 Saved to: {output_csv}")
    
    return {
        'method': 'OCR (img2table)',
        'time': processing_time,
        'tables_found': len(docs),
        'status': 'success',
        'dataframes': docs
    }

def main():
    src = "../data/pubtabnet/images/sample_00034.png"
    result = extract_tables_from_image(src, "TC03.csv")
    print(f"\n✅ Completed: {result['tables_found']} tables in {result['time']:.2f}s")


if __name__ == "__main__":
    main()
