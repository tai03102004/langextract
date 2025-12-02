import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import langextract as lx
import textwrap
import json  
import os 

app = FastAPI()

#Define Prompt 
prompt = textwrap.dedent("""
    Extract banking transaction information from Vietnamese SMS/notifications.
    Focus on identifying the account owner vs other party accounts.
    Use exact text for extractions. Do not paraphrase.
    Ensure all required fields are extracted when possible.
""")

# extraction information
def load_examples():
    examples_file = "../data/langextract-examples.json"
    if (os.path.exists(examples_file)):
        with open(examples_file, 'r', encoding='utf-8') as f:
            examples_data = json.load(f)
        examples = []
        for ex in examples_data:
            examples.append(
                lx.data.ExampleData( 
                    text=ex["text"],
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=e["extraction_class"],
                            extraction_text=e["extraction_text"],
                            attributes=e.get("attributes", {})
                        ) for e in ex["extractions"]
                    ]
                )
            )
        return examples
    else:
        raise FileNotFoundError(f"Examples file not found: {examples_file}")

def load_examples_pdf():
    example_files = "../data/Table_2.pdf"
    if (os.path.exists(example_files)):
        with open(example_files, 'rb') as f:
            file_data = f.read()
        file_b64 = base64.b64encode(file_data).decode("utf-8")
        examples = [
            lx.data.ExampleData(
                text="PDF Document",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="Document",
                        extraction_text="Table_2.pdf",
                        attributes={"file_data": file_b64}
                    )
                ]
            )
        ]
        return examples
    else:
        raise FileNotFoundError(f"Examples file not found: {example_files}")

promptPdf = """
Extract all tables in this scientific paper into structured JSON.
Each table should include headers and rows clearly separated.
"""
def test_langextract_pdf():
    examples = load_examples_pdf()
    result = lx.extract(
        text_or_documents="Please analyze the attached PDF document.",
        prompt_description=promptPdf,
        examples=examples,
        model_id="qwen2.5:3b",
        model_url="http://localhost:11434",
    )
    for r in result.extractions:
        print("="*60)
        print("Extraction class:", r.extraction_class)
        print("Extraction text:", r.extraction_text[:300])
        print("Attributes:", r.attributes)
        
# Input model cho request
# class TransactionInput(BaseModel):
#     text: str
#     model_id: str = "gemini-2.5-flash"

# API endpoint để NodeJS gọi
# @app.post("/extract")
# def extract_transaction(data: TransactionInput):
#     try:
#         examples = load_examples()
#         if data.model_id.startswith("gemini"):
#             # Use Gemini API
#             result = lx.extract(
#                 text_or_documents=data.text,
#                 prompt_description=prompt,
#                 examples=examples, 
#                 model_id=data.model_id,
#                 api_key=os.getenv("GEMINI_API_KEY", "AIzaSyAqc0swRcgIqeXO7492Qmn5oX6PZyBrma0")
#             )
#         else:
#             # Use local Ollama
#             result = lx.extract(
#                 text_or_documents=data.text,
#                 prompt_description=prompt,
#                 examples=examples,
#                 model_id=data.model_id,
#                 model_url="http://localhost:11434"
#             )
#         extractions = [
#             {
#                 "extraction_class": r.extraction_class,
#                 "extraction_text": r.extraction_text,
#                 "attributes": r.attributes or {}
#             }
#             for r in result.extractions
#         ]
#         return {
#             "success": True,
#             "model_used": data.model_id,
#             "results": extractions
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "service": "langextract"}

if __name__ == "__main__":

    test_langextract_pdf()

    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)