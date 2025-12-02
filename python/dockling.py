
import logging
import time
from pathlib import Path

import pandas as pd

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode


_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../data"
    input_doc_path = data_folder / "Table_1.pdf"

    output_dir = Path("scratch")

    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],  
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    if not conv_res.document.tables:
        print("⚠️ Không tìm thấy bảng nào trong PDF")
    else:
        print(f"✅ Tổng số bảng phát hiện: {len(conv_res.document.tables)}")

    doc_filename = conv_res.input.file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    full_content_md = conv_res.document.export_to_markdown()
    full_content_path = output_dir / f"{doc_filename}-full_content.md"
    with full_content_path.open("w", encoding="utf-8") as f:
        f.write(full_content_md)

    _log.info(f"Full document content saved to {full_content_path}")

    # if conv_res.document.tables:
    #     print(f"✅ Tổng số bảng phát hiện: {len(conv_res.document.tables)}")
    #     for idx, table in enumerate(conv_res.document.tables):
    #         df: pd.DataFrame = table.export_to_dataframe()
    #         table_md = table.export_to_markdown()

    #         table_md_path = output_dir / f"{doc_filename}-table-{idx + 1}.md"
    #         with table_md_path.open("w", encoding="utf-8") as f:
    #             f.write(table_md)

    #         csv_path = output_dir / f"{doc_filename}-table-{idx + 1}.csv"
    #         df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    #         _log.info(f"Table {idx + 1} saved to {table_md_path} and {csv_path}")
    # else:
    #     print("⚠️ Không tìm thấy bảng nào trong PDF")

    end_time = time.time() - start_time
    _log.info(f"Hoàn thành tất cả trong {end_time:.2f} giây.")


if __name__ == "__main__":
    main()