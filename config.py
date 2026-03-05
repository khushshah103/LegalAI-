import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Verified models based on diagnostic output
# Using 2.0-flash as primary for speed, pro-latest for complex tasks
PRIMARY_MODEL = "gemini-2.0-flash"
COMPLEX_MODEL = "gemini-pro-latest"

# Fallback chain for reliability
MODEL_FALLBACKS = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash"  # Experimental but available
]

# --- App Settings ---
APP_NAME = "LegalAI Portable"
MAX_CHAR_LIMIT = 60000  # Streamlit/API context window limit
ENTITY_EXTRACTION_CHARS = 35000  # Smart sampling limit

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FINE_TUNED_MODEL_PATH = os.path.join(BASE_DIR, "legal_bert_finetuned_risk")

def is_model_loaded():
    return GEMINI_API_KEY is not None and GEMINI_API_KEY != ""
