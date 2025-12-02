import logging
import time
from pathlib import Path
import pandas as pd
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image
import os
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

_log = logging.getLogger(__name__)
def display_file(file_path):
    """
    Display a PDF file or an image file.

    Args:
        file_path (str): Path to the PDF or image file.
    """
    # Check file extension to determine the type
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        # Convert and display PDF pages as images
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            plt.figure(figsize=(16, 12))
            plt.imshow(image)
            plt.axis('off')  # Hide axis for a cleaner look
            plt.show()
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Open and display the image file
        image = Image.open(file_path)
        plt.figure(figsize=(16, 12))
        plt.imshow(image)
        plt.axis('off')  # Hide axis for a cleaner look
        plt.show()
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or image file.")
    
def extract_data_with_docling(input_data_path):
    """
    Extracts data from a PDF or image file using the Docling library. 
    Displays the document's content as markdown and exports any tables found in the document to CSV files.

    Args:
        input_data_path (str): The path to the input file (PDF or image).
    """
    # Initialize pipeline options with table structure analysis enabled
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model

    # Create a document converter with specified format options
    doc_converter = DocumentConverter(
        allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
            ],  # whitelist formats, non-matching files are ignored.
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = doc_converter.convert(input_data_path)
    print(result.document.export_to_markdown())

    doc_filename = result.input.file.stem
    # Loop through each table detected in the document and Export it
    for table_ix, table in enumerate(result.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        #print(table_df.to_markdown())

        # Save the table as csv
        element_csv_filename = f"{doc_filename}-table-{table_ix+1}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_df.to_csv(element_csv_filename)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pdf_path = "../data/Table_2.pdf"
    
    display_file(pdf_path)
    extract_data_with_docling(pdf_path)
