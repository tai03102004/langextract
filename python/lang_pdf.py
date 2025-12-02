import os
import pandas as pd
import pdfplumber # type: ignore
from typing import List, Dict, Any

def extract_tables_to_csv(pdf_path: str, output_dir: str):
    """
    Trích xuất tất cả bảng từ PDF và lưu trực tiếp thành file CSV.
    Đây là cách làm hiệu quả và chính xác nhất.
    """
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file PDF tại: {pdf_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Tạo thư mục output tại: {output_dir}")
    
    csv_counter = 1

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Đang xử lý PDF: {pdf_path}...")
        for page_num, page in enumerate(pdf.pages, start=1):
            
            # Dùng hàm gốc của pdfplumber
            tables = page.extract_tables()
            
            if not tables:
                print(f"--- Trang {page_num}: Không tìm thấy bảng ---")
                continue
            
            print(f"--- Trang {page_num}: Tìm thấy {len(tables)} bảng ---")
            
            for t_idx, table_data in enumerate(tables, start=1):
                if not table_data or len(table_data) < 2:
                    print(f"   -> Bảng {t_idx} rỗng hoặc chỉ có header, bỏ qua.")
                    continue
                
                try:
                    # Dữ liệu đã sạch ngay từ đây
                    headers = table_data[0]
                    data_rows = table_data[1:]
                    
                    # Tạo DataFrame trực tiếp
                    df = pd.DataFrame(data_rows, columns=headers)
                    
                    # Lưu CSV
                    csv_path = os.path.join(output_dir, f"native_p{page_num}_t{t_idx}.csv")
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    print(f"   -> 💾 Đã lưu CSV thành công: {csv_path}")
                    csv_counter += 1
                except Exception as e:
                    print(f"   -> ⚠️ Lỗi khi xử lý bảng {t_idx} trang {page_num}: {e}")

def main():
    extract_tables_to_csv(
        pdf_path="../data/Table_2.pdf", 
        output_dir="output_tables_NATIVE" # Đặt tên khác để so sánh
    )
    
if __name__ == "__main__":
    main()