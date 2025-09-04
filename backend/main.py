from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF

app = FastAPI()

# Allow frontend requests (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-pdf/")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF"""
    text = ""
    with fitz.open(stream=await file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return {"text": text.strip()}
