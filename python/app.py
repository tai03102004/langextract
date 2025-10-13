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
    

# Input model cho request
class TransactionInput(BaseModel):
    text: str
    model_id: str = "gemini-2.5-flash"

# API endpoint để NodeJS gọi
@app.post("/extract")
def extract_transaction(data: TransactionInput):
    try:
        examples = load_examples()
        if data.model_id.startswith("gemini"):
            # Use Gemini API
            result = lx.extract(
                text_or_documents=data.text,
                prompt_description=prompt,
                examples=examples,
                model_id=data.model_id,
                api_key=os.getenv("GEMINI_API_KEY", "AIzaSyAqc0swRcgIqeXO7492Qmn5oX6PZyBrma0")
            )
        else:
            # Use local Ollama
            result = lx.extract(
                text_or_documents=data.text,
                prompt_description=prompt,
                examples=examples,
                model_id=data.model_id,
                model_url="http://localhost:11434"
            )
        extractions = [
            {
                "extraction_class": r.extraction_class,
                "extraction_text": r.extraction_text,
                "attributes": r.attributes or {}
            }
            for r in result.extractions
        ]
        return {
            "success": True,
            "model_used": data.model_id,
            "results": extractions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "langextract"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)