"""
Benchmark hoàn chỉnh: Chạy 3 phương pháp + Đánh giá accuracy
"""
import sys
import time
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to import các file
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import 3 phương pháp
from lang_pdf import extract_tables_from_pdf as langextract_method
from extract_table import extract_tables_ocr as ocr_method  
from dockling import extract_tables_docling as docling_method
from table_transformer import extract_tables_transformer as transformer_method  


# Import accuracy evaluator
from dataset.accuracy_evaluator import AccuracyEvaluator


class CompleteBenchmark:
    """Benchmark đầy đủ với accuracy evaluation"""
    
    def __init__(self, 
                 data_dir="../../data",
                 predictions_dir="../../data/predictions"):
        """
        Args:
            data_dir: Thư mục chứa PDFs và ground truth
            predictions_dir: Thư mục lưu kết quả predictions
        """
        self.data_dir = Path(data_dir)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.test_cases_dir = self.data_dir / "test_cases"
        self.predictions_dir = Path(predictions_dir)
        
        # Tạo thư mục predictions cho từng method
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
        for method in ['langextract', 'ocr', 'docling', 'transformer']:
            (self.predictions_dir / method).mkdir(exist_ok=True)
        
        # Initialize accuracy evaluator
        self.accuracy_evaluator = AccuracyEvaluator(
            ground_truth_dir=str(self.test_cases_dir)
        )
        
        self.results = []
    
    def get_test_cases_from_metadata(self) -> list:
        """
        Load test cases từ metadata CSV
        
        Returns:
            List of dicts: [{'id': 'TC01', 'pdf_path': '...', 'ground_truth_path': '...'}, ...]
        """
        metadata_path = self.test_cases_dir / "test_cases_metadata.csv"
        
        if not metadata_path.exists():
            print(f"❌ Metadata not found: {metadata_path}")
            return []
        
        metadata_df = pd.read_csv(metadata_path)
        test_cases = []
        
        for _, row in metadata_df.iterrows():
            tc_id = row['id']
            pdf_file = row['pdf_file']
            
            # PDF path
            pdf_path = self.pdfs_dir / pdf_file
            
            # Ground truth path
            gt_path = self.test_cases_dir / f"{tc_id}_ground_truth.csv"
            
            # Check if both files exist
            if pdf_path.exists() and gt_path.exists():
                test_cases.append({
                    'id': tc_id,
                    'pdf_path': str(pdf_path),
                    'ground_truth_path': str(gt_path),
                    'pdf_file': pdf_file
                })
            else:
                if not pdf_path.exists():
                    print(f"⚠️  PDF not found: {pdf_path}")
                if not gt_path.exists():
                    print(f"⚠️  Ground truth not found: {gt_path}")
        
        return test_cases
    
    def run_extraction_and_save(self, test_case_id: str, pdf_path: str, method: str) -> dict:
        """
        Chạy extraction và lưu kết quả vào predictions/method/
        
        Args:
            test_case_id: ID của test case (TC01, TC02, ...)
            pdf_path: Path đến PDF file
            method: Tên phương pháp (langextract, ocr, docling)
            
        Returns:
            Dict với keys: method, time, tables_found, status, dataframes
        """
        # Output path: predictions/method/TC01.csv
        output_csv = self.predictions_dir / method / f"{test_case_id}.csv"
        
        print(f"\n{'='*60}")
        print(f"🔬 Method: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            if method == 'langextract':
                result = langextract_method(pdf_path, str(output_csv))
            elif method == 'ocr':
                result = ocr_method(pdf_path, str(output_csv))
            elif method == 'docling':
                result = docling_method(pdf_path, str(output_csv))
            elif method == 'transformer': 
                result = transformer_method(pdf_path, str(output_csv))
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {
                'method': method,
                'time': 0,
                'tables_found': 0,
                'status': f'error: {str(e)}',
                'dataframes': []
            }
    
    def run_full_benchmark(self, test_cases: list = None) -> pd.DataFrame:
        """
        Chạy benchmark đầy đủ
        
        Args:
            test_cases: List of test cases (nếu None, load từ metadata)
            
        Returns:
            DataFrame với kết quả benchmark
        """
        if test_cases is None:
            test_cases = self.get_test_cases_from_metadata()
        
        if not test_cases:
            print("❌ No test cases found!")
            return pd.DataFrame()
        
        methods = ['langextract', 'ocr', 'docling', 'transformer']
        all_results = []
        
        print("="*80)
        print("🚀 BẮT ĐẦU COMPLETE BENCHMARK")
        print("="*80)
        print(f"\n📋 Test cases: {len(test_cases)}")
        for tc in test_cases:
            print(f"   - {tc['id']}: {tc['pdf_file']}")
        
        for tc in test_cases:
            tc_id = tc['id']
            pdf_path = tc['pdf_path']
            
            print(f"\n{'='*80}")
            print(f"📄 Test Case: {tc_id}")
            print(f"   PDF: {tc['pdf_file']}")
            print(f"   Ground Truth: {Path(tc['ground_truth_path']).name}")
            print(f"{'='*80}")
            
            for method in methods:
                # 1. Extraction + Timing
                extraction_result = self.run_extraction_and_save(tc_id, pdf_path, method)
                
                # 2. Accuracy evaluation
                predicted_csv = self.predictions_dir / method / f"{tc_id}.csv"
                
                if predicted_csv.exists():
                    print(f"\n   📊 Evaluating accuracy...")
                    accuracy_result = self.accuracy_evaluator.evaluate_method_on_testcase(
                        tc_id, str(predicted_csv)
                    )
                else:
                    print(f"   ⚠️  No predicted CSV found: {predicted_csv}")
                    accuracy_result = {
                        'cell_accuracy': None,
                        'structure_accuracy': None,
                        'shape_match': False,
                        'total_cells': 0,
                        'correct_cells': 0
                    }
                
                # 3. Combine results
                combined_result = {
                    'test_case_id': tc_id,
                    'method': method,
                    'processing_time': extraction_result['time'],
                    'tables_found': extraction_result['tables_found'],
                    'status': extraction_result['status'],
                    'cell_accuracy': accuracy_result.get('cell_accuracy'),
                    'structure_accuracy': accuracy_result.get('structure_accuracy'),
                    'shape_match': accuracy_result.get('shape_match', False),
                    'total_cells': accuracy_result.get('total_cells', 0),
                    'correct_cells': accuracy_result.get('correct_cells', 0),
                    'avg_similarity': accuracy_result.get('avg_similarity', 0)
                }
                
                all_results.append(combined_result)
                
                # Print summary
                print(f"\n   📊 SUMMARY:")
                print(f"      ⏱️  Time: {extraction_result['time']:.2f}s")
                print(f"      📊 Tables: {extraction_result['tables_found']}")
                print(f"      ✅ Status: {extraction_result['status']}")
                if accuracy_result.get('cell_accuracy') is not None:
                    print(f"      🎯 Cell Accuracy: {accuracy_result['cell_accuracy']:.1f}%")
                    print(f"      🏗️  Structure Accuracy: {accuracy_result['structure_accuracy']:.1f}%")
                    print(f"      📐 Shape Match: {accuracy_result['shape_match']}")
        
        # Save all results
        df = pd.DataFrame(all_results)
        output_csv = self.predictions_dir / "complete_benchmark_results.csv"
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 Saved detailed results to: {output_csv}")
        
        return df
    
    def create_final_report(self, df: pd.DataFrame):
        """
        Tạo báo cáo tổng kết
        """
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)
        
        # Summary by method
        summary = df.groupby('method').agg({
            'processing_time': ['mean', 'std', 'min', 'max'],
            'tables_found': ['sum', 'mean'],
            'cell_accuracy': ['mean', 'std'],
            'structure_accuracy': ['mean', 'std'],
            'avg_similarity': ['mean']
        }).round(2)
        
        print("\n### Summary by Method:")
        print(summary)
        
        # Save summary
        summary.to_csv(self.predictions_dir / "final_summary.csv", encoding='utf-8-sig')
        
        # Create comparison table for report
        methods_list = df['method'].unique()
        comparison_data = {
            'Tiêu chí': [
                'Thời gian TB (s)', 
                'Độ chính xác Cell TB (%)', 
                'Độ chính xác Structure TB (%)', 
                'Similarity TB (%)',
                'Tổng bảng phát hiện'
            ]
        }
        
        for method in methods_list:
            method_data = df[df['method'] == method]
            comparison_data[method] = [
                round(method_data['processing_time'].mean(), 2),
                round(method_data['cell_accuracy'].mean(), 2) if method_data['cell_accuracy'].notna().any() else 0,
                round(method_data['structure_accuracy'].mean(), 2) if method_data['structure_accuracy'].notna().any() else 0,
                round(method_data['avg_similarity'].mean(), 2) if method_data['avg_similarity'].notna().any() else 0,
                int(method_data['tables_found'].sum())
            ]
        
        comparison_table = pd.DataFrame(comparison_data)
        
        print("\n### Comparison Table for Report:")
        print(comparison_table.to_string(index=False))
        
        # Save as CSV
        comparison_table.to_csv(self.predictions_dir / "comparison_table.csv", index=False, encoding='utf-8-sig')
        
        # Save as LaTeX
        latex_table = comparison_table.to_latex(index=False)
        with open(self.predictions_dir / "comparison_table.tex", 'w') as f:
            f.write(latex_table)
        
        # Create visualizations
        self.create_visualizations(df)
        
        print(f"\n💾 Saved final report to: {self.predictions_dir}/")
        print(f"   Files created:")
        print(f"   - complete_benchmark_results.csv (detailed)")
        print(f"   - final_summary.csv")
        print(f"   - comparison_table.csv")
        print(f"   - comparison_table.tex")
        print(f"   - comparison_charts.png")
        print(f"   - accuracy_heatmap.png")
    
    def create_visualizations(self, df: pd.DataFrame):
        """Tạo biểu đồ"""
        # Chart 1: Time & Accuracy comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1.1 Processing Time
        time_data = df.groupby('method')['processing_time'].mean().sort_values()
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        time_data.plot(kind='barh', ax=axes[0], color=colors[:len(time_data)])
        axes[0].set_xlabel('Thời gian TB (giây)')
        axes[0].set_title('So sánh thời gian xử lý')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 1.2 Cell Accuracy
        acc_data = df.groupby('method')['cell_accuracy'].mean().dropna()
        if not acc_data.empty:
            acc_data.plot(kind='bar', ax=axes[1], color=colors[:len(acc_data)])
            axes[1].set_ylabel('Độ chính xác (%)')
            axes[1].set_title('Độ chính xác Cell')
            axes[1].set_xticklabels(acc_data.index, rotation=45, ha='right')
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].set_ylim([0, 100])
        
        # 1.3 Tables Found
        tables_data = df.groupby('method')['tables_found'].sum()
        tables_data.plot(kind='bar', ax=axes[2], color=colors[:len(tables_data)])
        axes[2].set_ylabel('Số bảng')
        axes[2].set_title('Tổng số bảng phát hiện')
        axes[2].set_xticklabels(tables_data.index, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.predictions_dir / "comparison_charts.png", dpi=300, bbox_inches='tight')
        print(f"   📊 Created: comparison_charts.png")
        
        # Chart 2: Accuracy Heatmap
        if df['cell_accuracy'].notna().any():
            plt.figure(figsize=(1, 6))
            pivot = df.pivot_table(
                values='cell_accuracy', 
                index='test_case_id', 
                columns='method'
            )
            
            if not pivot.empty:
                sns.heatmap(
                    pivot, 
                    annot=True, 
                    fmt='.1f', 
                    cmap='YlGnBu', 
                    cbar_kws={'label': 'Độ chính xác (%)'},
                    vmin=0,
                    vmax=100
                )
                plt.title('Heatmap: Độ chính xác Cell theo Test Case')
                plt.tight_layout()
                plt.savefig(self.predictions_dir / "accuracy_heatmap.png", dpi=300, bbox_inches='tight')
                print(f"   📊 Created: accuracy_heatmap.png")
        
        plt.close('all')


def main():
    """Main function"""
    
    # Initialize benchmark
    benchmark = CompleteBenchmark(
        data_dir="../../data",
        predictions_dir="../../data/predictions"
    )
    
    # Run full benchmark (auto load từ metadata)
    df = benchmark.run_full_benchmark()
    
    if not df.empty:
        # Create report
        benchmark.create_final_report(df)
        
        print("\n" + "="*80)
        print("✅ COMPLETE BENCHMARK FINISHED!")
        print("="*80)
        print(f"\n📁 Results location:")
        print(f"   {benchmark.predictions_dir}")
    else:
        print("\n❌ Benchmark failed - no results generated")


if __name__ == "__main__":
    main()