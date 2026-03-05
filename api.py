import os
import torch
import re
import json
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from google import genai
from dotenv import load_dotenv

from src.improved_extractor import ImprovedExtractor
from src.rag_service import RAGService
from io import BytesIO
import config

# Load environment variables
load_dotenv()

app = FastAPI(title="LegalAI API", version="1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
classifier = None
client = None
extractor = None
rag = None

def load_services():
    global classifier, client, extractor, rag
    
    # 1. Risk Classifier
    torch.manual_seed(42)
    local_path = config.FINE_TUNED_MODEL_PATH
    model_to_load = "nlpaueb/legal-bert-base-uncased"
    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        model_to_load = local_path
        
    try:
        classifier = pipeline(
            "text-classification", 
            model=model_to_load,
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        print(f"✅ API: Classifier loaded using {model_to_load}")
    except Exception as e:
        print(f"⚠️ API: Classifier failed: {e}")

    # 2. Gemini LLM (Centralized)
    if config.GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            print("✅ API: Gemini Client initialized")
        except Exception as e:
            print(f"⚠️ API: Gemini Client failed: {e}")
    else:
        # Fallback to Vertex if config exists but key doesn't
        vertex_json = os.path.join(config.BASE_DIR, "vertex_config.json")
        if os.path.exists(vertex_json):
            try:
                with open(vertex_json, "r") as f:
                    v_config = json.load(f)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(vertex_json)
                client = genai.Client(vertexai=True, project=v_config.get("project_id"), location="us-central1")
                print("✅ API: Vertex AI Client initialized")
            except Exception as e:
                print(f"⚠️ API: Vertex AI failed: {e}")

    # 3. Extractor
    extractor = ImprovedExtractor()

    # 4. RAG Service
    rag = RAGService()
    try:
        rag_data_path = os.path.join(config.BASE_DIR, "data", "text")
        if os.path.exists(rag_data_path):
            rag.load_documents(rag_data_path)
            print(f"✅ API: RAG loaded from {rag_data_path}")
    except Exception as e:
        print(f"⚠️ API: RAG loading failed: {e}")

@app.on_event("startup")
async def startup_event():
    load_services()


# ====================================================================
# Helper: Call Gemini with standardized fallback
# ====================================================================
def call_gemini(prompt, preferred_model=None):
    """Single helper for all Gemini calls. Retries with backoff and model fallback."""
    if not client:
        print("❌ API: No Gemini client available")
        return None
    
    # Use preferred model if provided, else fallback to config list
    models_to_try = [preferred_model] if preferred_model else config.MODEL_FALLBACKS
    
    for attempt in range(2): # 2 main retry loops
        for model_id in models_to_try:
            try:
                # Basic generation
                response = client.models.generate_content(model=model_id, contents=prompt)
                if response and response.text:
                    return response.text
                else:
                    print(f"⚠️ API: {model_id} returned empty response")
            except Exception as e:
                err = str(e).upper()
                # Handle Rate Limits
                if "429" in err or "QUOTA" in err or "LIMIT" in err:
                    wait = 10 * (attempt + 1)
                    print(f"⏳ API: Rate limited on {model_id}. Waiting {wait}s...")
                    time.sleep(wait)
                    break # Try next model or next attempt
                # Handle Model Not Found (404)
                elif "404" in err or "NOT FOUND" in err:
                    print(f"❌ API: Model {model_id} NOT FOUND. Skipping.")
                    continue # Try next model in list
                else:
                    print(f"❌ API: {model_id} failed: {e}")
                    continue
    return None


# ====================================================================
# Utility
# ====================================================================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'_{2,}', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip()


# ====================================================================
# API Models
# ====================================================================
from pydantic import BaseModel
from typing import List, Optional

class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    text: str
    history: List[dict]
    prompt: str

class ComplianceRequest(BaseModel):
    text: str
    framework: str

class SearchRequest(BaseModel):
    query: str


# ====================================================================
# ENDPOINTS (each defined ONCE, clean and simple)
# ====================================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "classifier": classifier is not None,
            "gemini": client is not None,
            "rag": rag is not None
        }
    }


@app.post("/api/analyze/extract")
async def extract_document(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    temp_path = f"temp_{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        raw_text = extractor.extract_text(temp_path)
        cleaned_text = clean_text(raw_text)
        return {"raw_text": raw_text, "cleaned_text": cleaned_text}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze/summary")
async def analyze_summary(req: TextRequest):
    prompt = f"""
    Act as a Senior Legal Counsel. Provide a professional, detailed, and structured executive summary.
    Use bold headings for: Purpose & Overview, Key Obligations, Payment & Compensation, Term & Termination, and Liability & Risk.
    
    Contract Text:
    {req.text[:60000]}
    """
    result = call_gemini(prompt)
    return {"summary": result or "⚠️ Summary generation failed."}


@app.post("/api/analyze/entities")
async def analyze_entities(req: TextRequest):
    # Smart sampling: first 20k + last 15k to capture preamble AND signature blocks
    first_part = req.text[:20000]
    last_part = req.text[-15000:] if len(req.text) > 20000 else ""
    sample = first_part + "\n\n--- END OF DOCUMENT ---\n\n" + last_part
    
    prompt = f"""
    Act as a Legal Clerk. Extract core legal entities from this contract.
    
    CRITICAL RULES:
    - Do NOT extract placeholders like [PROVIDER LEGAL NAME] or [CUSTOMER].
    - Check the signature block at the end for actual names.
    - If only placeholders exist, write "NOT SPECIFIED (Generic Template)".
    
    Extract:
    1. Contracting Parties (Full legal names)
    2. Effective Date
    3. Governing Law
    4. Total Contract Value
    
    Contract Text:
    {sample}
    """
    result = call_gemini(prompt)
    return {"entities_text": result or "⚠️ Entity extraction failed."}


@app.post("/api/analyze/scam")
async def analyze_scam(req: TextRequest):
    prompt = f"""
    Act as a Senior Contract Auditor. Scan for predatory, hidden, or highly imbalanced clauses.
    Focus on: IP transfers, uncapped liability, sneaky auto-renewals, hidden exit fees.
    
    If found, respond: FLAGGED: [1-sentence explanation]
    If safe, respond: SAFE
    
    Contract Text:
    {req.text[:60000]}
    """
    result = call_gemini(prompt)
    if result and "FLAGGED:" in result:
        return {"scam_warning": result.split("FLAGGED:")[1].strip()}
    return {"scam_warning": None}


@app.post("/api/analyze/risk")
async def analyze_risk(req: TextRequest):
    if not classifier:
        return {"label": "N/A", "score": 0.0, "description": ""}
    try:
        cleaned = clean_text(req.text)
        result = classifier(cleaned[:512])[0]
        label_id = result['label']
        mapping = {
            "LABEL_0": ("High Risk", "Critical issues found. Requires legal review."), 
            "LABEL_1": ("Low Risk", "Standard safe clauses. Low legal overhead."), 
            "LABEL_2": ("Medium Risk", "Minor deviations found. Proceed with caution.")
        }
        name, desc = mapping.get(label_id, (label_id, ""))
        return {"label": name, "score": result['score'], "description": desc}
    except Exception as e:
        return {"label": f"Error: {e}", "score": 0.0, "description": ""}


@app.post("/api/analyze/compliance")
async def analyze_compliance(req: ComplianceRequest):
    prompt = f"""
    Act as an expert compliance auditor. Check this contract against: '{req.framework}'.
    
    Evaluate 4-5 critical requirements. For each, give Pass (✅) or Fail (❌) with a 1-sentence reason.
    Format as a clean Markdown list.
    
    Contract Text:
    {req.text[:60000]}
    """
    result = call_gemini(prompt)
    return {"result": result or "⚠️ Compliance check failed."}


@app.post("/api/analyze/compare")
async def analyze_compare(req: TextRequest):
    prompt = f"""
    Extract the following strictly as a JSON object with EXACTLY these keys:
    "Vendor_Name", "Total_Pricing", "Term_Duration", "Liability_Cap", "Termination_Notice"
    
    If a field is missing, use "Not Specified".
    Do NOT output any markdown blocks. ONLY output the raw JSON object.

    Contract text:
    {req.text[:15000]}
    """
    result = call_gemini(prompt)
    if result:
        try:
            resp_text = result.strip()
            if resp_text.startswith("```json"):
                resp_text = resp_text[7:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            data = json.loads(resp_text)
            return {"data": data}
        except Exception:
            pass
    return {"data": None}


@app.post("/api/chat")
async def chat_document(req: ChatRequest):
    history_text = ""
    for msg in req.history:
        role_str = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role_str}: {msg['content']}\n"
        
    prompt = f"""
    You are a helpful legal assistant. Answer based ONLY on the contract text.
    If the answer is not in the text, say "I cannot find the answer to this in the document."
    
    Contract Text:
    {req.text[:100000]}
    
    Conversation History:
    {history_text}
    
    Latest Question: {req.prompt}
    """
    result = call_gemini(prompt)
    return {"answer": result or "Failed to generate answer."}


@app.post("/api/library/search")
async def library_search(req: SearchRequest):
    if not rag:
        return {"error": "RAG service not loaded"}
    try:
        relevant_chunks = rag.query(req.query, top_k=3)
        answer = rag.generate_answer(req.query, relevant_chunks, client)
        refs = [{"file": res["metadata"]["file"], "text": res["text"]} for res in relevant_chunks]
        return {"answer": answer, "references": refs}
    except Exception as e:
        return {"error": f"Search failed: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
