import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import re

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("⚠️  python-Levenshtein not installed. Using simple string comparison.")


class AccuracyEvaluator:
    """Đánh giá độ chính xác"""
    
    def __init__(self, ground_truth_dir="../data/test_cases"):
        self.ground_truth_dir = Path(ground_truth_dir)
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. Make column names unique
        cols = []
        col_counts = {}
        for col in df.columns:
            col_clean = re.sub(r'[^\w\s]', '', str(col).strip().lower())
            
            # Add suffix nếu duplicate
            if col_clean in col_counts:
                col_counts[col_clean] += 1
                col_clean = f"{col_clean}_{col_counts[col_clean]}"
            else:
                col_counts[col_clean] = 0
            
            cols.append(col_clean)
        
        df.columns = cols
        
        # 2. Convert to string and strip
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        
        # 3. Replace null values
        df = df.replace(['nan', 'None', 'NaN', 'null'], '')
        
        # 4. Check if completely empty
        all_empty = df.apply(lambda col: col.astype(str).str.strip().eq('').all()).all()
        if all_empty:
            print(f"   ⚠️  DataFrame is completely empty after normalization")
            return pd.DataFrame()
        
        # 5. Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def align_dataframes(self, gt_df: pd.DataFrame, ex_df: pd.DataFrame) -> tuple:
        gt_df = self.normalize_dataframe(gt_df)
        ex_df = self.normalize_dataframe(ex_df)
        
        alignment_info = {
            'gt_shape': gt_df.shape,
            'ex_shape': ex_df.shape,
            'columns_match': False,
            'rows_diff': abs(gt_df.shape[0] - ex_df.shape[0]),
            'cols_diff': abs(gt_df.shape[1] - ex_df.shape[1])
        }
        
        # Align columns first
        gt_cols = gt_df.shape[1]
        ex_cols = ex_df.shape[1]
        
        if list(gt_df.columns) == list(ex_df.columns):
            alignment_info['columns_match'] = True
        elif gt_cols == ex_cols:
            # Same count, force GT headers
            ex_df.columns = gt_df.columns
            alignment_info['columns_match'] = True
            alignment_info['column_alignment'] = 'forced_by_count'
        
        # Pad or truncate columns to match GT
        if ex_cols < gt_cols:
            # Add missing columns with empty values
            for i in range(ex_cols, gt_cols):
                col_name = gt_df.columns[i]
                ex_df[col_name] = ''
            print(f"   ⚠️  Added {gt_cols - ex_cols} empty columns to extracted")
        elif ex_cols > gt_cols:
            # Truncate extra columns
            ex_df = ex_df.iloc[:, :gt_cols]
            print(f"   ⚠️  Truncated {ex_cols - gt_cols} extra columns")
        
        # Ensure column order matches
        ex_df = ex_df[gt_df.columns]
        
        # Pad or truncate rows to match GT
        gt_rows = gt_df.shape[0]
        ex_rows = ex_df.shape[0]
        
        if ex_rows < gt_rows:
            # Add missing rows with empty values
            missing_rows = gt_rows - ex_rows
            empty_rows = pd.DataFrame(
                [['' for _ in range(gt_cols)] for _ in range(missing_rows)],
                columns=gt_df.columns
            )
            ex_df = pd.concat([ex_df, empty_rows], ignore_index=True)
            print(f"   ⚠️  Added {missing_rows} empty rows to extracted")
        elif ex_rows > gt_rows:
            # Truncate extra rows (penalize for extra data)
            ex_df = ex_df.iloc[:gt_rows]
            print(f"   ⚠️  Truncated {ex_rows - gt_rows} extra rows")
        
        # Final shape check
        assert gt_df.shape == ex_df.shape, f"Shape mismatch after alignment: GT={gt_df.shape}, EX={ex_df.shape}"
        
        return gt_df, ex_df, alignment_info
        
    def calculate_cell_similarity(self, gt_val: str, ex_val: str) -> float:
        """
        Tính similarity giữa 2 cell values
        
        Returns:
            Float from 0.0 to 1.0
        """
        gt_val = str(gt_val).strip().lower()
        ex_val = str(ex_val).strip().lower()
        
        # Empty cells
        if gt_val == '' and ex_val == '':
            return 1.0
        
        # Exact match
        if gt_val == ex_val:
            return 1.0
        
        # Levenshtein similarity
        if HAS_LEVENSHTEIN and gt_val and ex_val:
            distance = Levenshtein.distance(gt_val, ex_val)
            max_len = max(len(gt_val), len(ex_val), 1)
            similarity = 1 - (distance / max_len)
            return similarity
        
        # Fallback: check if one contains the other
        if gt_val in ex_val or ex_val in gt_val:
            return 0.7
        
        return 0.0
    
    def calculate_cell_accuracy(self, 
                                ground_truth_df: pd.DataFrame, 
                                extracted_df: pd.DataFrame) -> Dict:
        """
        Tính độ chính xác từng cell với alignment
        """
        # Align dataframes
        gt_df, ex_df, alignment_info = self.align_dataframes(ground_truth_df, extracted_df)
        
        if gt_df.empty or ex_df.empty:
            return {
                'cell_accuracy': 0.0,
                'shape_match': False,
                'total_cells': 0,
                'correct_cells': 0,
                'avg_similarity': 0.0,
                'exact_matches': 0,
                'partial_matches': 0,
                'alignment_info': alignment_info
            }
        
        # So sánh từng cell
        total_cells = gt_df.size
        exact_matches = 0
        partial_matches = 0  # similarity > 0.8
        similarity_scores = []
        
        for col in gt_df.columns:
            for idx in gt_df.index:
                gt_val = str(gt_df.loc[idx, col]).strip()
                ex_val = str(ex_df.loc[idx, col]).strip()
                
                similarity = self.calculate_cell_similarity(gt_val, ex_val)
                similarity_scores.append(similarity)
                
                if similarity == 1.0:
                    exact_matches += 1
                elif similarity > 0.8:
                    partial_matches += 1
        
        # Calculate weighted correct cells
        correct_cells = exact_matches + (partial_matches * 0.5)
        
        return {
            'cell_accuracy': (correct_cells / total_cells) * 100 if total_cells > 0 else 0.0,
            'shape_match': gt_df.shape == ex_df.shape,
            'total_cells': total_cells,
            'correct_cells': correct_cells,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'avg_similarity': np.mean(similarity_scores) * 100 if similarity_scores else 0.0,
            'alignment_info': alignment_info
        }
    
    def calculate_structure_accuracy(self, 
                                     ground_truth_df: pd.DataFrame, 
                                     extracted_df: pd.DataFrame) -> Dict:
        """
        Đánh giá độ chính xác cấu trúc bảng
        """
        gt_df = self.normalize_dataframe(ground_truth_df)
        ex_df = self.normalize_dataframe(extracted_df)
        
        metrics = {
            'row_count_match': gt_df.shape[0] == ex_df.shape[0],
            'col_count_match': gt_df.shape[1] == ex_df.shape[1],
            'header_match': list(gt_df.columns) == list(ex_df.columns),
            'gt_shape': str(gt_df.shape),
            'extracted_shape': str(ex_df.shape),
            'row_diff': abs(gt_df.shape[0] - ex_df.shape[0]),
            'col_diff': abs(gt_df.shape[1] - ex_df.shape[1])
        }
        
        # Tính điểm tổng với trọng số
        structure_score = (
            metrics['row_count_match'] * 0.4 +  # 40% for rows
            metrics['col_count_match'] * 0.4 +  # 40% for columns
            metrics['header_match'] * 0.2       # 20% for headers
        ) * 100
        
        metrics['structure_accuracy'] = structure_score
        
        return metrics
    
    def evaluate_method_on_testcase(self, 
                                test_case_id: str, 
                                extracted_csv_path: str) -> Dict:
        gt_path = self.ground_truth_dir / f"{test_case_id}_ground_truth.csv"
        
        if not gt_path.exists():
            print(f"   ❌ Ground truth not found: {gt_path}")
            return {
                'test_case_id': test_case_id,
                'cell_accuracy': 0.0,
                'structure_accuracy': 0.0,
                'shape_match': False,
                'error': 'ground_truth_not_found'
            }
        
        if not Path(extracted_csv_path).exists():
            print(f"   ❌ Extracted result not found: {extracted_csv_path}")
            return {
                'test_case_id': test_case_id,
                'cell_accuracy': 0.0,
                'structure_accuracy': 0.0,
                'shape_match': False,
                'error': 'extracted_result_not_found'
            }
        
        try:
            # Read CSVs
            try:
                gt_df = pd.read_csv(gt_path, encoding='utf-8-sig')
            except:
                gt_df = pd.read_csv(gt_path, encoding='utf-8')
            
            try:
                extracted_df = pd.read_csv(extracted_csv_path, encoding='utf-8-sig')
            except:
                extracted_df = pd.read_csv(extracted_csv_path, encoding='utf-8')
            
            print(f"   📊 Ground Truth: {gt_df.shape}")
            print(f"   📊 Extracted: {extracted_df.shape}")
            
            # Check completely empty
            if gt_df.empty and extracted_df.empty:
                print(f"   ⚠️  Both DataFrames are empty")
                return {
                    'test_case_id': test_case_id,
                    'cell_accuracy': 100.0,  # Empty vs empty = perfect match
                    'structure_accuracy': 100.0,
                    'shape_match': True,
                    'total_cells': 0,
                    'correct_cells': 0.0,
                    'error': 'both_empty'
                }
            
            if gt_df.empty:
                # GT empty but extracted has data = wrong
                return {
                    'test_case_id': test_case_id,
                    'cell_accuracy': 0.0,
                    'structure_accuracy': 0.0,
                    'shape_match': False,
                    'total_cells': 0,
                    'correct_cells': 0.0,
                    'error': 'gt_empty_but_extracted_not'
                }
            
            if extracted_df.empty:
                # GT has data but extracted empty = all wrong
                total_cells = gt_df.shape[0] * gt_df.shape[1]
                return {
                    'test_case_id': test_case_id,
                    'cell_accuracy': 0.0,
                    'structure_accuracy': 0.0,
                    'shape_match': False,
                    'total_cells': total_cells,
                    'correct_cells': 0.0,
                    'error': 'extracted_empty'
                }
            
            # Calculate metrics
            cell_metrics = self.calculate_cell_accuracy(gt_df, extracted_df)
            structure_metrics = self.calculate_structure_accuracy(gt_df, extracted_df)
            
            print(f"   ✅ Cell Accuracy: {cell_metrics['cell_accuracy']:.1f}%")
            print(f"   ✅ Exact matches: {cell_metrics['exact_matches']}/{cell_metrics['total_cells']}")
            print(f"   ✅ Partial matches: {cell_metrics['partial_matches']}")
            
            return {
                'test_case_id': test_case_id,
                **cell_metrics,
                **structure_metrics
            }
            
        except Exception as e:
            print(f"   ❌ Error evaluating {test_case_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'test_case_id': test_case_id,
                'cell_accuracy': 0.0,
                'structure_accuracy': 0.0,
                'shape_match': False,
                'error': str(e)
            }