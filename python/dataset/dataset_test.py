"""
Tạo bộ test dataset với ground truth để đánh giá
"""
import pandas as pd
import os
from pathlib import Path

class TestDatasetCreator:
    """Tạo và quản lý test dataset"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.test_cases_dir = self.data_dir / "test_cases"
        self.test_cases_dir.mkdir(exist_ok=True)
    
    def create_test_cases_metadata(self):
        """
        Tạo metadata mô tả từng test case
        """
        test_cases = [
            {
                'id': 'TC01',
                'name': 'Simple Table',
                'pdf_file': 'TC01_sample1.pdf',
                'description': 'Bảng đơn giản; có border rõ ràng; không merge cell',
                'complexity': 'Low',
                'expected_tables': 2,
                'table_features': ['clear_borders', 'single_header', 'no_merge']
            },
            {
                'id': 'TC02',
                'name': 'Simple Table with Line Breaks',
                'pdf_file': 'TC02_Simple_Table_LineBreaks.pdf',
                'description': 'Bảng đơn giản; có border rõ ràng; không merge cell; các ô có xuống dòng',
                'complexity': 'Low',
                'expected_tables': 1,
                'table_features': ['clear_borders', 'single_header', 'no_merge', 'line_breaks']
            },
            {
                'id': 'TC03',
                'name': 'Complex Table with Merge Cells and long',
                'pdf_file': 'TC03_Complex_Table_MergeCells_Long.pdf',
                'description': 'Bảng phức tạp; có merge cells; multi-level headers; bảng dài',
                'complexity': 'High',
                'expected_tables': 1,
                'table_features': ['merged_cells', 'multi_level_headers', 'long_table']
            },
            {
                'id': 'TC04',
                'name': 'Borderless Table',
                'pdf_file': 'TC04_Table_No_Borders.pdf',
                'description': 'Bảng không có đường kẻ; chỉ dựa vào khoảng trắng',
                'complexity': 'Medium',
                'expected_tables': 1,
                'table_features': ['no_borders', 'whitespace_separation']
            },
            {
                'id': 'TC05',
                'name': 'Table with Special Characters and Empty Cells',
                'pdf_file': 'TC05_Table_SpecialChars_EmptyCells.pdf',
                'description': 'Bảng có các ký tự đặc biệt; có nhiều ô rỗng',
                'complexity': 'Medium',
                'expected_tables': 1,
                'table_features': ['special_characters', 'empty_cells']
            },
            {
                'id': 'TC06',
                'name': 'Scanned PDF',
                'pdf_file': 'TC06_scanned_table.pdf',
                'description': 'PDF scan từ ảnh; chất lượng trung bình',
                'complexity': 'High',
                'expected_tables': 1,
                'table_features': ['scanned_image', 'low_quality']
            },
            {
                'id': 'TC07',
                'name': 'Long Table without Borders on Multiple Pages',
                'pdf_file': 'TC07_table_no_borders_multi_page.pdf',
                'description': 'Không có border; nhiều trang',
                'complexity': 'Medium',
                'expected_tables': 1,
                'table_features': ['long_table', 'multi_page', 'no_borders']
            }
        ]

        for tc in test_cases:
            tc["table_features"] = " | ".join(tc["table_features"])
        
        df = pd.DataFrame(test_cases)
        csv_path = self.test_cases_dir / "test_cases_metadata.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Created test cases metadata: {csv_path}")
        return df
    
    def create_ground_truth_for_table(self, test_case_id: str, table_data: list):
        """
        Tạo ground truth CSV cho từng bảng
        
        Args:
            test_case_id: ID của test case (TC01, TC02, ...)
            table_data: List of lists chứa dữ liệu bảng đúng
        """
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        
        gt_path = self.test_cases_dir / f"{test_case_id}_ground_truth.csv"
        df.to_csv(gt_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Created ground truth: {gt_path}")
        return gt_path

def create_sample_ground_truths():
    """Tạo ground truth mẫu"""
    creator = TestDatasetCreator()

    creator.create_test_cases_metadata()

if __name__ == "__main__":
    create_sample_ground_truths()