import json
import time
import ast
from pathlib import Path
import pandas as pd
import re
from difflib import SequenceMatcher
from tqdm import tqdm

from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions, TesseractOcrOptions

import signal

def parse_pubtabnet_annotation(annotation: dict) -> dict:
    """Parse PubTabNet annotation đúng format"""
    
    html_str = annotation.get('html', '')
    
    try:
        if isinstance(html_str, str):
            data = ast.literal_eval(html_str)
        else:
            data = html_str
    except:
        return {'cells': [], 'rows': 0, 'cols': 0, 'texts': [], 'has_merged': False}
    
    cells = data.get('cells', [])
    structure = data.get('structure', {}).get('tokens', [])
    
    # Extract text từ cells
    texts = []
    for cell in cells:
        tokens = cell.get('tokens', [])
        text = ''.join(tokens)
        text = re.sub(r'<[^>]+>', '', text)
        texts.append(text.strip())
    
    rows = structure.count('<tr>')
    
    structure_str = ' '.join(structure)
    has_colspan = 'colspan' in structure_str
    has_rowspan = 'rowspan' in structure_str
    
    return {
        'cells': cells,
        'rows': rows,
        'cell_count': len(cells),
        'texts': texts,
        'has_merged': has_colspan or has_rowspan
    }


def normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh - cải thiện"""
    if not text:
        return ""
    text = text.strip().lower()
    # Xóa ký tự đặc biệt nhưng giữ số và chữ
    text = re.sub(r'[^\w\s\.\-\+±%]', '', text)
    # Chuẩn hóa số thập phân
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)  # "0 . 5" -> "0.5"
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check nếu 2 text giống nhau >= threshold"""
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    
    if not t1 or not t2:
        return t1 == t2
    
    # Exact match
    if t1 == t2:
        return True
    
    # Fuzzy match
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio >= threshold


def calculate_accuracy(docling_df: pd.DataFrame, gt_info: dict) -> dict:
    """Tính accuracy giữa Docling output và ground truth"""
    
    docling_rows = docling_df.shape[0]
    docling_cols = docling_df.shape[1]
    docling_cells = docling_rows * docling_cols
    
    gt_rows = gt_info['rows']
    gt_cells = gt_info['cell_count']
    gt_texts = gt_info['texts']
    
    # Row accuracy
    if gt_rows > 0:
        row_accuracy = min(docling_rows, gt_rows) / max(docling_rows, gt_rows) * 100
    else:
        row_accuracy = 0
    
    # Cell accuracy
    if gt_cells > 0:
        cell_accuracy = min(docling_cells, gt_cells) / max(docling_cells, gt_cells) * 100
    else:
        cell_accuracy = 0
    
    # Extract texts từ Docling DataFrame
    docling_texts = []
    for r in range(docling_rows):
        for c in range(docling_cols):
            val = docling_df.iloc[r, c]
            text = str(val) if pd.notna(val) else ""
            if text and text != 'nan':
                docling_texts.append(text)
    
    # Normalize texts
    docling_texts_norm = [normalize_text(t) for t in docling_texts if t.strip()]
    gt_texts_norm = [normalize_text(t) for t in gt_texts if t.strip()]
    
    # ========== TEXT ACCURACY - SET-BASED ==========
    # Đếm số text trong Docling mà có match trong GT (fuzzy match)
    
    if docling_texts_norm and gt_texts_norm:
        # Exact match count
        exact_matches = 0
        fuzzy_matches = 0
        
        gt_matched = set()  # Track GT texts đã được match
        
        for dt in docling_texts_norm:
            best_match_ratio = 0
            best_match_idx = -1
            
            for idx, gt in enumerate(gt_texts_norm):
                if idx in gt_matched:
                    continue
                    
                ratio = SequenceMatcher(None, dt, gt).ratio()
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_match_idx = idx
            
            if best_match_ratio == 1.0:
                exact_matches += 1
                gt_matched.add(best_match_idx)
            elif best_match_ratio >= 0.5:  # 70% similar = fuzzy match
                fuzzy_matches += 1
                gt_matched.add(best_match_idx)
        
        # Text accuracy = % GT texts được tìm thấ
        # y trong Docling
        total_matches = exact_matches + fuzzy_matches
        text_accuracy = (total_matches / len(gt_texts_norm)) * 100
        exact_match_rate = (exact_matches / len(gt_texts_norm)) * 100
        fuzzy_match_rate = (fuzzy_matches / len(gt_texts_norm)) * 100
        
    else:
        text_accuracy = 0
        exact_match_rate = 0
        fuzzy_match_rate = 0
    
    return {
        'row_accuracy': row_accuracy,
        'cell_accuracy': cell_accuracy,
        'text_accuracy': text_accuracy,
        'exact_match_rate': exact_match_rate,
        'fuzzy_match_rate': fuzzy_match_rate,
        'gt_rows': gt_rows,
        'gt_cells': gt_cells,
        'docling_rows': docling_rows,
        'docling_cols': docling_cols,
        'docling_cells': docling_cells,
        'has_merged': gt_info['has_merged'],
        'gt_texts': gt_texts[:5],
        'docling_texts': docling_texts[:5]
    }


def test_docling_on_pubtabnet(num_samples: int = 10):
    """Test Docling trên PubTabNet samples"""
    
    images_dir = Path("../../data/pubtabnet/images")
    annotations_dir = Path("../../data/pubtabnet/annotations")
    output_dir = Path("../../data/output/docling_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(images_dir.glob("*.png"))[1:num_samples+1]

    print("=" * 85)
    print("🔍 Test Docling on PubTabNet - SET-BASED Accuracy Evaluation")
    print("=" * 85)
    
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
        }
    )

    print("\n⏳ Warming up model...")
    warmup_img = list(images_dir.glob("*.png"))[0]
    _ = doc_converter.convert(str(warmup_img))
    print("✅ Model loaded!")

    print("\n⏳ Loading models...")
    results = []
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        print(f"\n📄 [{idx+1}/{len(image_files)}] {img_path.name}")
        
        start_time = time.time()
        
        try:
            conv_res = doc_converter.convert(str(img_path))
            processing_time = time.time() - start_time
            
            anno_path = annotations_dir / f"{img_path.stem}.json"
            with open(anno_path) as f:
                annotation = json.load(f)
            
            gt_info = parse_pubtabnet_annotation(annotation)
            
            if conv_res.document.tables:
                for table in conv_res.document.tables:
                    df = table.export_to_dataframe()
                    
                    csv_path = output_dir / f"{img_path.stem}.csv"
                    df.to_csv(csv_path, index=False)
                    
                    acc = calculate_accuracy(df, gt_info)
                    
                    print(f"   ⏱️  Time: {processing_time:.2f}s")
                    print(f"   📊 Docling: {acc['docling_rows']}r × {acc['docling_cols']}c = {acc['docling_cells']} cells")
                    print(f"   📋 GT:      {acc['gt_rows']}r, {acc['gt_cells']} cells")
                    print(f"   ✅ Row Acc:     {acc['row_accuracy']:.1f}%")
                    print(f"   ✅ Cell Acc:    {acc['cell_accuracy']:.1f}%")
                    print(f"   ✅ Text Acc:    {acc['text_accuracy']:.1f}% (Exact: {acc['exact_match_rate']:.1f}% + Fuzzy: {acc['fuzzy_match_rate']:.1f}%)")
                    
                    results.append({
                        'file': img_path.name,
                        'time': processing_time,
                        'docling_rows': acc['docling_rows'],
                        'docling_cols': acc['docling_cols'],
                        'docling_cells': acc['docling_cells'],
                        'gt_rows': acc['gt_rows'],
                        'gt_cells': acc['gt_cells'],
                        'row_accuracy': acc['row_accuracy'],
                        'cell_accuracy': acc['cell_accuracy'],
                        'text_accuracy': acc['text_accuracy'],
                        'exact_match_rate': acc['exact_match_rate'],
                        'fuzzy_match_rate': acc['fuzzy_match_rate'],
                        'has_merged': acc['has_merged'],
                        'status': 'success'
                    })
            else:
                print(f"   ❌ No table detected")
                results.append({
                    'file': img_path.name,
                    'time': processing_time,
                    'docling_rows': 0,
                    'docling_cols': 0,
                    'docling_cells': 0,
                    'gt_rows': gt_info['rows'],
                    'gt_cells': gt_info['cell_count'],
                    'row_accuracy': 0,
                    'cell_accuracy': 0,
                    'text_accuracy': 0,
                    'exact_match_rate': 0,
                    'fuzzy_match_rate': 0,
                    'has_merged': gt_info['has_merged'],
                    'status': 'no_table'
                })
                    
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ❌ Error: {str(e)[:80]}")
            
            results.append({
                'file': img_path.name,
                'time': processing_time,
                'docling_rows': 0,
                'docling_cols': 0,
                'docling_cells': 0,
                'gt_rows': 0,
                'gt_cells': 0,
                'row_accuracy': 0,
                'cell_accuracy': 0,
                'text_accuracy': 0,
                'exact_match_rate': 0,
                'fuzzy_match_rate': 0,
                'has_merged': False,
                'status': 'error'
            })
    
    # Summary
    print("\n" + "=" * 85)
    print("📊 SUMMARY")
    print("=" * 85)
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'File':<18} {'Docling':<10} {'GT':<12} {'Row%':<7} {'Cell%':<7} {'Text%':<7} {'Exact%':<7}")
    print("-" * 85)
    
    for _, r in df_results.iterrows():
        docling_str = f"{r['docling_rows']}×{r['docling_cols']}"
        gt_str = f"{r['gt_rows']}r,{r['gt_cells']}c"
        print(f"{r['file']:<18} {docling_str:<10} {gt_str:<12} {r['row_accuracy']:<7.1f} {r['cell_accuracy']:<7.1f} {r['text_accuracy']:<7.1f} {r['exact_match_rate']:<7.1f}")
    
    success_df = df_results[df_results['status'] == 'success']
    
    if len(success_df) > 0:
        print("\n" + "=" * 85)
        print("📈 OVERALL STATISTICS")
        print("=" * 85)
        
        print(f"\n✅ Success rate: {len(success_df)}/{len(df_results)} ({len(success_df)/len(df_results)*100:.1f}%)")
        
        if len(success_df) > 1:
            avg_time = success_df['time'].iloc[1:].mean()
        else:
            avg_time = success_df['time'].mean()
        print(f"⏱️  Avg time: {avg_time:.2f}s")
        
        print(f"\n📊 Average Accuracy:")
        print(f"   Row Accuracy:    {success_df['row_accuracy'].mean():.1f}%")
        print(f"   Cell Accuracy:   {success_df['cell_accuracy'].mean():.1f}%")
        print(f"   Text Accuracy:   {success_df['text_accuracy'].mean():.1f}%")
        print(f"   Exact Match:     {success_df['exact_match_rate'].mean():.1f}%")
        print(f"   Fuzzy Match:     {success_df['fuzzy_match_rate'].mean():.1f}%")
    
    df_results.to_csv(output_dir / "accuracy_report_v2.csv", index=False)
    print(f"\n💾 Saved to: {output_dir}/accuracy_report_v2.csv")
    
    return results


if __name__ == "__main__":
    test_docling_on_pubtabnet(num_samples=200)