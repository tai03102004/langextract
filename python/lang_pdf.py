import os
import time
import pandas as pd
from pathlib import Path
import pdfplumber
import requests


def extract_tables_from_pdf(pdf_path: str, output_csv: str = None) -> dict:
    """
    Extract tables from PDF: pdfplumber (text) → Ollama LLM (structure) → DataFrame
    """
    print(f"🤖 Extracting tables from: {pdf_path}")
    start_time = time.time()
    
    try:
        # 1. Extract text from PDF
        print("   📄 Extracting text...")
        all_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        
        combined_text = "\n\n".join(all_text)
        print(f"   ✅ Extracted {len(combined_text)} chars")
        
        if not combined_text.strip():
            raise ValueError("No text found in PDF")
        
        # 2. Call Ollama LLM
        print("   🔄 Processing with LLM...")
        prompt = f"""Extract ALL tables from this document. Return ONLY pipe-separated tables.

        Rules:
        1. Each table must have a header row
        2. Use | as column separator
        3. One row per line
        4. No markdown, no explanations
        5. If cell is empty, write "EMPTY"
        6. Preserve all rows and columns
        Text:
        {combined_text[:4000]}

        Format:
        Header1 | Header2 | Header3
        Value1 | Value2 | Value3

        Return tables only, no explanations."""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        result_text = response.json()['response']
        processing_time = time.time() - start_time
        
        # 3. Parse pipe-separated tables
        print("   📊 Parsing tables...")
        lines = [line.strip() for line in result_text.split('\n') 
                 if line.strip() and '|' in line]
        
        dataframes = []
        if len(lines) >= 2:
            rows = []
            for line in lines:
                
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty first/last
                while cells and cells[0] == '':
                    cells.pop(0)
                while cells and cells[-1] == '':
                    cells.pop()
                
                if cells and len(cells) > 1:
                    rows.append(cells)
            
            if len(rows) >= 2:
                # Create DataFrame
                headers = rows[0]
                data_rows = rows[1:]
                
                # Ensure all rows have same columns
                clean_rows = []
                for row in data_rows:
                    if len(row) == len(headers):
                        clean_rows.append(row)
                    elif len(row) < len(headers):
                        clean_rows.append(row + [''] * (len(headers) - len(row)))
                    else:
                        clean_rows.append(row[:len(headers)])
                
                if clean_rows:
                    df = pd.DataFrame(clean_rows, columns=headers)
                    dataframes.append(df)
                    print(f"   ✅ Table: {df.shape}")
        
        # 4. Save CSV
        if output_csv and dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"   💾 Saved: {output_csv}")
        
        return {
            'method': 'LangExtract',
            'time': processing_time,
            'tables_found': len(dataframes),
            'status': 'success',
            'dataframes': dataframes
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ❌ Error: {str(e)}")
        return {
            'method': 'LangExtract',
            'time': processing_time,
            'tables_found': 0,
            'status': f'error: {str(e)}',
            'dataframes': []
        }


def test_langextract():
    """Test function"""
    pdf_path = "../data/pdfs/TC01_sample1.pdf"
    output_csv = "langextract_output.csv"
    
    print("="*80)
    print("🧪 TESTING LANGEXTRACT")
    print("="*80)
    
    result = extract_tables_from_pdf(pdf_path, output_csv)
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"⏱️  Time: {result['time']:.2f}s")
    print(f"📊 Tables: {result['tables_found']}")
    print(f"✅ Status: {result['status']}")
    
    if result['dataframes']:
        for i, df in enumerate(result['dataframes'], 1):
            print(f"\n📋 Table {i}:")
            print(df.to_string(index=False))


if __name__ == "__main__":
    test_langextract()