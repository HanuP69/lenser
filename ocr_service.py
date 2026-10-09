import io
from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image

app = FastAPI()


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes))
    extracted_text = pytesseract.image_to_string(image)
    return {"text": extracted_text.strip()}


@app.get("/health")
def health_endpoint() -> dict[str, str]:
    return {"status": "ok"}
