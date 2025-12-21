import logging
import time
from pathlib import Path
import pandas as pd

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

_log = logging.getLogger(__name__)


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


def extract_tables_docling(pdf_path: str, output_csv: str = None) -> dict:

    start_time = time.time()
    
    try:
        # 1. Configure pipeline
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # 2. Convert PDF
        conv_res = doc_converter.convert(pdf_path)
        
        processing_time = time.time() - start_time
        
        dataframes = []
        
        # 3. Extract tables
        if conv_res.document.tables:
            for i, table in enumerate(conv_res.document.tables, 1):
                df = table.export_to_dataframe()
                if df.empty or df.shape[0] == 0:
                    print(f"   ⚠️  Table {i}: Empty, skipping")
                    continue
                df = make_columns_unique(df)
                
                dataframes.append(df)
                print(f"   📊 Table {i}: {df.shape}")
        else:
            print("   ⚠️  No tables found")
        
        # 4. Save to CSV if requested
        if output_csv and dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"   💾 Saved to: {output_csv}")
        
        return {
            'method': 'Docling TableFormer',
            'time': processing_time,
            'tables_found': len(dataframes),
            'status': 'success',
            'dataframes': dataframes
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ❌ Error: {str(e)}")
        
        return {
            'method': 'Docling TableFormer',
            'time': processing_time,
            'tables_found': 0,
            'status': f'error: {str(e)}',
            'dataframes': []
        }

def main():
    logging.basicConfig(level=logging.INFO)
    
    data_folder = Path(__file__).parent / "../data/pdfs"
    input_doc_path = data_folder / "TC05_Table_SpecialChars_EmptyCells.pdf"
    
    result = extract_tables_docling(str(input_doc_path), "TC05.csv")
    
    print(f"\n✅ Completed: {result['tables_found']} tables in {result['time']:.2f}s")


if __name__ == "__main__":
    main()