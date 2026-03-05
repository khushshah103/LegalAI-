import os
import fitz  # PyMuPDF
from .ocr_processor import OCRProcessor

class ImprovedExtractor:
    def __init__(self):
        self.ocr_processor = None # Lazy load OCR only if needed
        
    def extract_text(self, pdf_path):
        """
        Attempts to extract text using PyMuPDF first.
        If extracted text is insufficient, falls back to OCR.
        """
        print(f"\n--- Extracting from: {os.path.basename(pdf_path)} ---")
        doc = fitz.open(pdf_path)
        text = ""
        num_pages = len(doc)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        
        # Heuristic: If we have less than 50 characters per page on average, 
        # it might be a scanned PDF or contain only images.
        avg_chars_per_page = len(text.strip()) / num_pages if num_pages > 0 else 0
        
        if avg_chars_per_page < 50:
            print(f"Warning: Low text content found ({avg_chars_per_page:.1f} chars/pg).")
            print("Falling back to OCR...")
            if self.ocr_processor is None:
                self.ocr_processor = OCRProcessor()
            text = self.ocr_processor.extract_text_from_pdf(pdf_path)
        else:
            print(f"Success: Extracted {len(text)} characters using text-based extraction.")
            
        return text

    def process_directory(self, directory, output_dir=None):
        if output_dir is None:
            output_dir = directory
            
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            output_path = os.path.join(output_dir, f"{pdf_file}.txt")
            
            extracted_text = self.extract_text(pdf_path)
            
            if extracted_text:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print(f"Saved extracted text to: {output_path}")
            else:
                print(f"Failed to extract text from: {pdf_file}")

if __name__ == "__main__":
    base_dir = r"c:\Users\User\Downloads\legal document analyzer"
    extractor = ImprovedExtractor()
    extractor.process_directory(base_dir)
