import streamlit as st
import os
import torch
from transformers import pipeline
from src.improved_extractor import ImprovedExtractor
from src.rag_service import RAGService
import re
import json
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="⚖️",
    layout="wide",
)

# Initialize Session State Globally with robust checks
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_file" not in st.session_state:
        st.session_state["current_file"] = None
    if "file_data" not in st.session_state:
        st.session_state["file_data"] = {}
    if "models_loaded" not in st.session_state:
        st.session_state["models_loaded"] = False

init_session_state()

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1e3a8a;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Global Services ---
@st.cache_resource(show_spinner=False)
def get_classifier(version=1):
    # Fix seed for consistent random head initialization (if fine-tuned path is missing)
    torch.manual_seed(42)
    
    # Check for local fine-tuned model from the user's notebook progress
    local_path = config.FINE_TUNED_MODEL_PATH
    model_to_load = "nlpaueb/legal-bert-base-uncased"
    
    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        model_to_load = local_path
        print(f"✅ Loading local fine-tuned model: {local_path}")
    else:
        print(f"ℹ️ Falling back to base Legal-BERT (random accuracy for risk).")
        
    return pipeline(
        "text-classification", 
        model=model_to_load,
        device=0 if torch.cuda.is_available() else -1,
        model_kwargs={"low_cpu_mem_usage": True}
    )

@st.cache_resource
def get_gemini_client():
    # Centralized client initialization
    if config.GEMINI_API_KEY:
        try:
            c = genai.Client(api_key=config.GEMINI_API_KEY)
            print("✅ Initialized Gemini Client")
            return c
        except Exception as e:
            print(f"❌ Gemini Client init failed: {e}")
    
    # Fallback to Vertex
    vertex_json = os.path.join(config.BASE_DIR, "vertex_config.json")
    if os.path.exists(vertex_json):
        try:
            with open(vertex_json, "r") as f:
                v_config = json.load(f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(vertex_json)
            c = genai.Client(
                vertexai=True,
                project=v_config.get("project_id"),
                location="us-central1"
            )
            print("✅ Initialized Vertex AI Client")
            return c
        except Exception as e:
            print(f"⚠️ Vertex AI failed: {e}")
    
    return None

# Removed BERT-based get_ner to save RAM. Using Gemini for Entity Extraction.

@st.cache_resource
def get_rag():
    r = RAGService()
    try:
        rag_data_path = os.path.join(config.BASE_DIR, "data", "text")
        if os.path.exists(rag_data_path):
            r.load_documents(rag_data_path)
    except Exception as e:
        print(f"Error loading RAG: {e}")
    return r

# --- UI Loader (Non-Cached) ---
def load_models(progress_bar=None, status_text=None):
    """
    Orchestrates the loading of all models with UI progress feedback.
    The individual load functions are cached, so this function run fast after the first time.
    """
    # 1. Risk Classifier
    if status_text: status_text.text("Loading Risk Classifier (BERT)...")
    if progress_bar: progress_bar.progress(20)
    classifier = get_classifier()

    # 2. Gemini LLM (for Summarization, RAG & NER)
    if status_text: status_text.text("Connecting to Gemini AI...")
    if progress_bar: progress_bar.progress(50)
    client = get_gemini_client()

    # 3. RAG Service
    if status_text: status_text.text("Indexing Document Library...")
    if progress_bar: progress_bar.progress(90)
    rag_service = get_rag()

    return {
        "classifier": classifier,
        "client": client,
        "extractor": ImprovedExtractor(),
        "rag": rag_service
    }

# --- Initial Setup & Startup Sequence ---
def startup_sequence():
    """
    Shows a splash screen only on the first run of the session.
    """
    if not st.session_state.get('models_loaded'):
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f"""
                <div style="text-align: center; padding: 50px;">
                    <h1 style="font-size: 3em;">⚖️ {config.APP_NAME}</h1>
                    <p style="font-size: 1.2em; color: #666;">Warming up the AI engine... This takes about 30-45 seconds on first boot.</p>
                </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # This triggers the individual cached loads
            load_models(progress_bar, status_text)
            
            progress_bar.progress(100)
            status_text.text("System ready!")
            st.session_state['models_loaded'] = True
            time.sleep(1)
        placeholder.empty()

# Run startup
startup_sequence()
models = load_models() # Returns instantly if already cached
classifier = models["classifier"]
client = models["client"]
extractor = models["extractor"]
rag = models["rag"]

def call_gemini(prompt, preferred_model=None):
    """Single helper for all Gemini calls. Retries with backoff and model fallback."""
    if not client:
        return None
    
    models_to_try = [preferred_model] if preferred_model else config.MODEL_FALLBACKS
    
    for attempt in range(2):
        for model_id in models_to_try:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                err = str(e).upper()
                if "429" in err or "QUOTA" in err or "LIMIT" in err:
                    if attempt < 1:
                        wait = 10 * (attempt + 1)
                        time.sleep(wait)
                        break
                    continue
                elif "404" in err or "NOT FOUND" in err:
                    continue
                else:
                    continue
    return None

def get_summary(text):
    prompt = f"""
    Act as a Senior Legal Counsel with 20 years of experience in contract law. 
    Review the provided legal contract and generate a high-level, professional executive summary.

    STRUCTURE YOUR RESPONSE AS FOLLOWS:
    1. **Executive Overview**: High-level purpose of the agreement.
    2. **Key Financial Terms**: Payment schedules, amounts, and late fees.
    3. **Operational Obligations**: What must each party actually DO?
    4. **Termination & Exit**: How do parties leave, and what are the notice periods?
    5. **Critical Liability & Risk**: Indemnities, liability caps, and any "one-sided" clauses.
    6. **Counsel's Recommendation**: A 2-3 sentence professional verdict on the contract's fairness.

    INSTRUCTIONS:
    - Use a professional, objective, and analytical tone.
    - If you find highly imbalanced or "predatory" clauses, mention them under "Critical Liability & Risk" in a factual, legal manner rather than using alarmist language.
    - Focus on specificities (dates, percentages, dollar amounts).
    - Ensure the summary is readable but dense with information.
    
    Contract Text:
    {text[:config.MAX_CHAR_LIMIT]}
    """
    result = call_gemini(prompt)
    return result or "⚠️ Summary generation failed."

def get_entities(text):
    """Smart sampling: first 20k + last 15k chars to capture preamble AND signature blocks."""
    first_part = text[:20000]
    last_part = text[-15000:] if len(text) > 20000 else ""
    sample = first_part + "\n\n--- END OF DOCUMENT ---\n\n" + last_part
    
    prompt = f"""
    Act as a Legal Clerk. Extract the following core entities from the contract.
    
    CRITICAL ACCURACY RULES:
    1. DO NOT extract generic placeholders like "[PROVIDER LEGAL NAME]", "[CUSTOMER]", or placeholders in curly brackets.
    2. Examine BOTH the introductory paragraph AND the signature blocks at the end for ACTUAL company names.
    3. If a field only contains a placeholder, write "NOT SPECIFIED (Generic Template Detected)".
    
    IDENTIFY:
    1. Contracting Parties (Full legal names of all parties involved)
    2. Effective Date (The start date of the agreement)
    3. Governing Law (Which state/country's laws apply)
    4. Total Contract Value (Specific monetary amount or fee structure)
    
    Contract Text:
    {sample}
    """
    result = call_gemini(prompt)
    return result or "⚠️ Entity extraction failed."

def check_unethical_clauses(text):
    """Scans full document (60k chars) for predatory clauses."""
    prompt = f"""
    Act as a Senior Contract Auditor. Scan for predatory, hidden, or highly imbalanced clauses.
    Focus on: IP transfers, uncapped liability, sneaky auto-renewals, hidden exit fees.
    
    If found, respond: FLAGGED: [1-sentence explanation]
    If safe, respond: SAFE
    
    Contract Text:
    {text[:60000]}
    """
    result = call_gemini(prompt)
    if result and "FLAGGED:" in result:
        return result.split("FLAGGED:")[1].strip()
    return None


def get_risk(text):
    if not classifier:
        return "N/A", 0.0, ""
    try:
        cleaned = clean_text(text)
        # Use first 512 tokens as BERT limit
        result = classifier(cleaned[:512])[0]
        label_id = result['label']
        mapping = {
            "LABEL_0": ("High Risk", "Critical issues found. Requires legal review."), 
            "LABEL_1": ("Low Risk", "Standard safe clauses. Low legal overhead."), 
            "LABEL_2": ("Medium Risk", "Minor deviations found. Proceed with caution.")
        }
        name, desc = mapping.get(label_id, (label_id, ""))
        return name, result['score'], desc
    except Exception as e:
        return f"Error: {e}", 0.0, ""



def run_compliance_check(text, framework):
    """Runs compliance audit against the full document (60k chars)."""
    prompt = f"""
    Act as an expert compliance auditor. Check this contract against: '{framework}'.
    
    Evaluate 4-5 critical requirements. For each, give Pass (✅) or Fail (❌) with 1-sentence reason.
    Format as a clean Markdown list.
    
    Contract Text:
    {text[:60000]}
    """
    result = call_gemini(prompt)
    return result or "⚠️ Compliance check failed."

def clean_text(text):
    """
    Cleans raw document text by removing noise like placeholders and extra whitespace.
    """
    if not text:
        return ""
    # Remove long sequences of underscores (placeholders)
    text = re.sub(r'_{2,}', '', text)
    # Remove curly bracket placeholders e.g. {services/project name}
    text = re.sub(r'\{.*?\}', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove "Page X" noise
    text = re.sub(r'Page \d+', '', text)
    return text.strip()
def get_comparison_data(text):
    prompt = f"""
    Extract the following strictly as a JSON object with EXACTLY these keys:
    "Vendor_Name", "Total_Pricing", "Term_Duration", "Liability_Cap", "Termination_Notice"
    
    If a field is missing, use "Not Specified".
    Do NOT output any markdown blocks. ONLY output the raw JSON object.

    Contract text:
    {text[:15000]}
    """
    result = call_gemini(prompt)
    if result:
        try:
            resp_text = result.strip()
            if resp_text.startswith("```json"):
                resp_text = resp_text[7:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            return json.loads(resp_text)
        except Exception:
            pass
    return None

# Check for model health
is_fine_tuned = os.path.isdir("legal_bert_finetuned_risk")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2901/2901306.png", width=100)
    st.title("Admin Panel")
    st.info("Upload your legal contracts to begin automated analysis. Upload multiple files for vendor comparison.")
    uploaded_files = st.file_uploader("Upload PDF Contract(s)", type=["pdf"], accept_multiple_files=True)

# --- Header ---
st.title("⚖️ Legal Document & Risk Analyzer")
st.markdown("Automated intelligence for contract review, risk mitigation, and semantic search.")

if not is_fine_tuned:
    st.warning("⚠️ **Warning:** No fine-tuned model found in `./legal_bert_finetuned_risk`. The risk classification is currently using 'Base' weights and will be inaccurate. Please provide your trained model files for accurate assessment.")

if uploaded_files:
    if len(uploaded_files) == 1:
        uploaded_file = uploaded_files[0]
        # Check if a new file was uploaded to reset the cache
        if st.session_state["current_file"] != uploaded_file.name:
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["messages"] = [] # Reset chat
            st.session_state["file_data"] = {} # Reset analytics cache

        # Save uploaded file to temp path
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Text Extraction
        with st.spinner("Processing document..."):
            raw_text = extractor.extract_text(uploaded_file.name)
            cleaned_text = clean_text(raw_text)
            
        # Clean up temp file immediately so it doesn't clutter the folder
        try:
            os.remove(uploaded_file.name)
        except OSError:
            pass

        # Phase 1: Summary & Scam Scan
        col_sum, col_risk = st.columns([2, 1])
        
        with col_sum:
            st.subheader("🤖 AI Summary")
            if "summary" not in st.session_state["file_data"]:
                with st.spinner("Analyzing document with Gemini 2.5 Pro..."):
                    st.session_state["file_data"]["summary"] = get_summary(raw_text)
            st.markdown(st.session_state["file_data"]["summary"])

            # Note: Scam/Ethical check is still performed but information is now integrated into the Risk and Summary sections
            if "scam_warning" not in st.session_state["file_data"]:
                time.sleep(2)  # Rate limit protection
                with st.spinner("Auditing clauses..."):
                    st.session_state["file_data"]["scam_warning"] = check_unethical_clauses(raw_text)

        with col_risk:
            st.subheader("⚖️ Risk Profile")
            if "risk" not in st.session_state["file_data"]:
                with st.spinner("Analyzing risk..."):
                    st.session_state["file_data"]["risk"] = get_risk(raw_text)
            
            label, score, description = st.session_state["file_data"]["risk"]
            color = "#16a34a" # Low
            if "Medium" in label: color = "#f59e0b"
            if "High" in label: color = "#ef4444"
            
            st.markdown(f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: white; text-align: center;">
                    <h2 style="color: white; margin: 0; font-size: 1.5em;">{label}</h2>
                </div>
                """, unsafe_allow_html=True)
            if description:
                st.markdown(f"**Actionable Insight:** {description}")
            if not is_fine_tuned:
                st.caption("🚨 Results are uncalibrated (Base Model).")
                    
        # Phase 2: Professional Entity Extraction
        st.divider()
        st.subheader("🔍 Key Legal Entities")
        if "entities" not in st.session_state["file_data"]:
            time.sleep(2)  # Rate limit protection
            with st.spinner("Extracting parties with AI..."):
                entities_text = get_entities(raw_text)  # Use raw_text for full context
                st.session_state["file_data"]["entities"] = entities_text if entities_text else "No entities detected."
        
        st.info(st.session_state["file_data"]["entities"])


        # Phase 2.5: Automated Compliance Checklists
        st.divider()
        st.subheader("🛡️ Compliance & Audit")
        st.markdown("Run automated checks against strict regulatory and industry frameworks.")
        
        frameworks = [
            "Select a framework to audit...",
            "General Data Protection Regulation (GDPR)",
            "Standard SaaS Agreement Best Practices",
            "Independent Contractor / Freelance Standard"
        ]
        
        selected_framework = st.selectbox("Select Compliance Framework:", frameworks)
        
        if selected_framework != "Select a framework to audit...":
            cache_key = f"compliance_{selected_framework}"
            
            if cache_key not in st.session_state["file_data"]:
                with st.spinner(f"Running {selected_framework} audit..."):
                    if not client:
                        st.warning("Gemini AI is not connected.")
                    else:
                        st.session_state["file_data"][cache_key] = run_compliance_check(cleaned_text, selected_framework)
            
            if st.session_state["file_data"].get(cache_key):
                st.info(st.session_state["file_data"][cache_key])
            elif client:
                st.error("Audit failed to generate.")

        # Phase 3: Interactive Chatbot
        st.divider()
        st.subheader("💬 Chat with Document")
        
        if not client:
            st.warning("⚠️ **Gemini API Key Missing:** Interactive chat is disabled.")
        else:
            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("Ask a question about this contract (e.g., 'What are the termination conditions?')..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Analyzing document and generating answer..."):
                        context = cleaned_text[:100000] # Safe limit for large documents
                        
                        # Build conversation history for the prompt
                        history_text = ""
                        for msg in st.session_state.messages[:-1]: # exclude the current prompt
                            role_str = "User" if msg["role"] == "user" else "Assistant"
                            history_text += f"{role_str}: {msg['content']}\n"
                            
                        full_prompt = f"""
                        You are a helpful legal assistant. Answer the user's latest question based ONLY on the following contract text and the conversation history so far. 
                        If the answer is not in the text, say "I cannot find the answer to this in the document."
                        
                        Contract Text:
                        {context}
                        
                        Conversation History:
                        {history_text}
                        
                        Latest Question: {prompt}
                        """
                        
                        answer = call_gemini(full_prompt)
                        if not answer:
                            answer = "Failed to generate answer. Please try again."
                        message_placeholder.markdown(answer)
                        
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # Multi-Document Vendor Comparison Mode
        st.header("📊 Multi-Document Vendor Comparison Matrix")
        st.markdown("Comparing key terms across multiple uploaded contracts.")
        
        comparison_results = []
        
        # We use a progress bar to show extraction status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Extracting data from {file.name} ({i+1}/{len(uploaded_files)})...")
            # Save temp
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
            
            # Extract
            raw_text = extractor.extract_text(file.name)
            cleaned_text = clean_text(raw_text)
            
            # Clean up temp file immediately
            try:
                os.remove(file.name)
            except OSError:
                pass
            
            # Get structured data
            data = get_comparison_data(cleaned_text)
            if data:
                data["Filename"] = file.name
                comparison_results.append(data)
            else:
                comparison_results.append({
                    "Filename": file.name,
                    "Vendor_Name": "Extraction Failed",
                    "Total_Pricing": "N/A",
                    "Term_Duration": "N/A",
                    "Liability_Cap": "N/A",
                    "Termination_Notice": "N/A"
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Extraction Complete!")
        
        if comparison_results:
            st.divider()
            df = pd.DataFrame(comparison_results)
            # Reorder columns to put Filename first
            cols = ["Filename", "Vendor_Name", "Total_Pricing", "Term_Duration", "Liability_Cap", "Termination_Notice"]
            df = df[[c for c in cols if c in df.columns]]
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.success("Comparison Matrix generated successfully! You can download this table via the download button inside the table view.")

else:
    st.warning("Please upload a PDF document in the sidebar to start analysis.")
    st.info("💡 **Tip:** You can use the Library Search below to query your repository of legal documents.")
    
    st.divider()
    st.subheader("📚 Global Library Search")
    lib_query = st.text_input("Identify patterns across your entire library:")
    if lib_query:
        with st.spinner("Searching library..."):
            relevant_chunks = rag.query(lib_query, top_k=3)
            answer = rag.generate_answer(lib_query, relevant_chunks, client)

            
            st.markdown("### 🤖 Synthesized Knowledge")
            st.success(answer)
            
            st.markdown("#### Document References")
            for i, res in enumerate(relevant_chunks):
                st.markdown(f"**{i+1}. From {res['metadata']['file']}:**")
                st.caption(res['text'])

# --- Footer ---
st.divider()
st.caption("LegalAI Analyzer v1.2 | Powered by Legal-BERT & T5")
