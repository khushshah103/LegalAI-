import os
import sys
import numpy as np
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io

class OCRProcessor:
    def __init__(self, languages=['en']):
        print(f"Initializing EasyOCR with languages: {languages}...")
        # Note: First run will download model weights (~100MB+)
        self.reader = easyocr.Reader(languages)
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Converts PDF pages to images using PyMuPDF and performs OCR on each page.
        """
        print(f"Opening PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return None
            
        full_text = ""
        for page_num in range(len(doc)):
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            page = doc.load_page(page_num)
            
            # Render page to a pixmap (image)
            # Zoom Factor (2.0 = 200% resolution for better OCR)
            zoom = 2.0 
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert PIL image to numpy array for EasyOCR
            page_np = np.array(img)
            
            # Perform OCR
            results = self.reader.readtext(page_np, detail=0)
            page_text = "\n".join(results)
            full_text += f"\n--- Page {page_num + 1} ---\n" + page_text + "\n"
            
        doc.close()
        return full_text

if __name__ == "__main__":
    # Test script
    test_pdf = sys.argv[1] if len(sys.argv) > 1 else None
    if test_pdf and os.path.exists(test_pdf):
        processor = OCRProcessor()
        text = processor.extract_text_from_pdf(test_pdf)
        if text:
            print("\nExtracted Text (first 500 chars):")
            print(text[:500])
        else:
            print("Extraction failed.")
    else:
        print("Usage: python ocr_processor.py <path_to_pdf>")
