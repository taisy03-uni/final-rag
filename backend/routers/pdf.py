from fastapi import APIRouter, UploadFile, File
import fitz  # PyMuPDF

router = APIRouter(prefix="/pdf", tags=["PDF"])

@router.post("/extract-pdf/")
async def extract_pdf(file: UploadFile = File(...)):
    text = ""
    with fitz.open(stream=await file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return {"text": text.strip()}
