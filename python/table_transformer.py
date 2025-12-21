import os
import io
import time
import pandas as pd
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import Image
from PIL import Image as PILImage

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    import torch
except ImportError:
    print("❌ Install: pip install transformers torch pillow pdf2image img2table pytesseract")
    exit(1)

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract/tessdata/'


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo tên cột không trùng"""
    cols = []
    col_counts = {}
    
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in col_counts:
            col_counts[col_str] += 1
            cols.append(f"{col_str}_{col_counts[col_str]}")
        else:
            col_counts[col_str] = 0
            cols.append(col_str)
    
    df.columns = cols
    return df


def clean_text(text):
    """Làm sạch text"""
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("\r", "").strip()


def extract_with_img2table(page_img: PILImage.Image, ocr: TesseractOCR, page_num: int) -> list:
    """Fallback extraction với img2table"""
    print(f"   ⚠️  Using img2table fallback...")
    
    img_bytes = io.BytesIO()
    page_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    doc = Image(img_bytes)
    
    # Thử bordered
    extracted = doc.extract_tables(
        ocr=ocr,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=30
    )
    
    # Thử borderless nếu không có
    if not extracted:
        extracted = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=20
        )
    
    results = []
    for idx, table_obj in enumerate(extracted, start=1):
        df = table_obj.df.copy()
        if df.empty:
            continue
        
        if len(df) > 0:
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        
        df = df.applymap(clean_text)
        df = make_columns_unique(df)
        
        results.append({
            'df': df,
            'info': {
                'page': page_num,
                'table_num': idx,
                'confidence': 0.0,
                'shape': df.shape,
                'method': 'fallback_img2table'
            }
        })
        
        print(f"      ✅ {df.shape[0]} rows × {df.shape[1]} cols")
    
    return results


def extract_tables_transformer(pdf_path: str, output_csv: str = None) -> dict:
    """
    Table Transformer với fallback
    - Detection model tìm vị trí bảng
    - img2table OCR nội dung
    - Fallback nếu không detect được
    """
    
    start_time = time.time()
    
    # Load model
    print("🔄 Loading Table Transformer...")
    try:
        processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        print("✅ Model loaded")
    except Exception as e:
        return {'method': 'Table Transformer', 'status': 'error', 'message': str(e)}
    
    # Convert PDF
    print(f"📄 Converting PDF...")
    pages = convert_from_path(pdf_path, dpi=400)
    print(f"   {len(pages)} pages")
    
    ocr = TesseractOCR(n_threads=2, lang="eng")
    docs = []
    all_tables_info = []
    
    # Process pages
    for page_num, page_img in enumerate(pages, start=1):
        print(f"\n🔍 Page {page_num}/{len(pages)}")
        
        # Convert to RGB
        if page_img.mode != 'RGB':
            page_img = page_img.convert('RGB')
        
        # Detect tables
        inputs = processor(images=page_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([page_img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, 
            threshold=0.5,
            target_sizes=target_sizes
        )[0]
        
        num_detected = len(results['scores'])
        print(f"   📊 Detected: {num_detected} tables")
        
        # Fallback nếu không detect được
        if num_detected == 0:
            fallback_results = extract_with_img2table(page_img, ocr, page_num)
            for res in fallback_results:
                docs.append(res['df'])
                all_tables_info.append(res['info'])
            continue
        
        # Process detected tables
        for idx, (score, bbox) in enumerate(zip(results["scores"], results["boxes"]), start=1):
            x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
            
            print(f"   🔲 Table {idx}: {score:.0%} confidence")
            
            # Crop và upscale
            table_crop = page_img.crop((x1, y1, x2, y2))
            w, h = table_crop.size
            table_crop = table_crop.resize((w * 2, h * 2), PILImage.LANCZOS)
            
            # OCR với img2table
            img_bytes = io.BytesIO()
            table_crop.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            
            doc = Image(img_bytes)
            extracted = doc.extract_tables(
                ocr=ocr,
                implicit_rows=False,
                implicit_columns=False,
                borderless_tables=False,
                min_confidence=30
            )
            
            if not extracted:
                extracted = doc.extract_tables(
                    ocr=ocr,
                    implicit_rows=True,
                    implicit_columns=True,
                    borderless_tables=True,
                    min_confidence=20
                )
            
            for table_obj in extracted:
                df = table_obj.df.copy()
                if df.empty:
                    continue
                
                if len(df) > 0:
                    df.columns = df.iloc[0]
                    df = df.iloc[1:].reset_index(drop=True)
                
                df = df.applymap(clean_text)
                df = make_columns_unique(df)
                
                docs.append(df)
                all_tables_info.append({
                    'page': page_num,
                    'table_num': idx,
                    'confidence': float(score),
                    'shape': df.shape,
                    'method': 'transformer'
                })
                
                print(f"      ✅ {df.shape[0]} rows × {df.shape[1]} cols")
    
    processing_time = time.time() - start_time
    
    # Save
    if output_csv and docs:
        combined_df = pd.concat(docs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n💾 Saved: {output_csv} ({len(combined_df)} rows)")
    
    return {
        'method': 'Table Transformer',
        'time': processing_time,
        'tables_found': len(docs),
        'status': 'success',
        'dataframes': docs,
        'tables_info': all_tables_info
    }


def main():
    pdf_path = "../data/pdfs/TC04_Table_No_Borders.pdf"
    output_csv = "TC03_transformer.csv"
    
    print("=" * 60)
    print("🚀 Table Transformer Extraction")
    print("=" * 60)
    
    result = extract_tables_transformer(pdf_path, output_csv)
    
    print("\n" + "=" * 60)
    print(f"✅ Done: {result['time']:.2f}s")
    print(f"📊 Tables: {result['tables_found']}")
    print("=" * 60)


if __name__ == "__main__":
    main()