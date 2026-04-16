import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from line_segmentation import detect_lines


def test_on_pdf_tables():
    """Test line detection trên PDF tables của bạn"""
    
    pdf_dir = Path("../../data/pdfs")
    output_dir = Path("../../data/debug_output")
    output_dir.mkdir(exist_ok=True)
    
    test_cases = [
        "TC01_sample1.pdf",
        "TC02_Simple_Table_LineBreaks.pdf",
        "TC03_Complex_Table_MergeCells_Long.pdf",
        "TC04_Table_No_Borders.pdf",
        "TC05_Table_SpecialChars_EmptyCells.pdf",
        "TC06_scanned_table.pdf",
        "TC07_Table_no_borders_multi_page.pdf",
    ]
    
    print("=" * 60)
    print("🔍 Line Segmentation Test - PDF Tables")
    print("=" * 60)
    
    results = []
    
    for tc in test_cases:
        pdf_path = pdf_dir / tc
        
        if not pdf_path.exists():
            print(f"\n❌ {tc}: File not found")
            continue
        
        # Convert PDF to image (first page only)
        pages = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
        img = np.array(pages[0])
        
        # Save temp image
        temp_path = output_dir / f"{tc.replace('.pdf', '.png')}"
        cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Detect lines
        result = detect_lines(str(temp_path))
        
        # Determine table type
        if result['horizontal'] >= 2 and result['vertical'] >= 2:
            detected_type = "bordered"
        else:
            detected_type = "borderless"
        
        print(f"\n📄 {tc}")
        print(f"   H-lines: {result['horizontal']}, V-lines: {result['vertical']}")
        print(f"   Type: {detected_type}")
        
        results.append({
            'file': tc,
            'h_lines': result['horizontal'],
            'v_lines': result['vertical'],
            'type': detected_type
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"{'File':<45} {'H':<5} {'V':<5} {'Type':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['file']:<45} {r['h_lines']:<5} {r['v_lines']:<5} {r['type']:<10}")


if __name__ == "__main__":
    test_on_pdf_tables()