---
title: LegalAI Analyzer
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# ⚖️ LegalAI Analyzer: AI-powered Contract Intelligence

[![Live Demo](https://img.shields.io/badge/Live-Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/khushshah103/LegalAI)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**LegalAI** is an AI-powered contract intelligence platform for automated contract review, structured summaries, risk detection and legal entity extraction. Features also include multi-document vendor comparison, conversational document chat and compliance auditing across major industry frameworks.

---

## 🚀 Why LegalAI instead of ChatGPT?

While generic LLMs are impressive, **LegalAI** offers specialized advantages for legal professionals and procurement teams:

| Feature | LegalAI Analyzer | Generic ChatGPT |
| :--- | :--- | :--- |
| **Legal Specialization** | Uses **Legal-BERT**, a transformer model trained exclusively on legal corpora. | General-purpose training; prone to legal "hallucinations." |
| **In-built Contract Chat** | Specialized **Document Chatbot** that maintains strict context of your legal files. | Requires constant copy-pasting; loses context in long threads. |
| **Industry Frameworks** | Pre-configured **Compliance Auditing** against major industry standards. | Requires manual prompting of every rule for every audit. |
| **RAG Precision** | Uses **Vector Search (FAISS)** to ensure answers are strictly grounded in *your* document. | Often draws on internal training data, leading to factual errors. |
| **Scanned Document Support** | Integrated **OCR (EasyOCR)** for processing scanned images and non-selectable PDFs. | Poor native support for "image-based" legal PDFs. |
| **Comparative Analysis** | Designed for **Vendor Comparison** (Multi-doc mode) with unified benchmarking. | Single-document focus; difficult to cross-compare. |

---

## 🔥 Key Features

- **🔍 Executive Summary**: Get a professional "Counsel's View" summary focusing on the core business impact.
- **⚠️ Risk Detection**: Automatically flag predatory clauses, uncapped liability, and imbalanced indemnity.
- **💬 Conversational Chat**: Query your documents directly—ask about notice periods, force majeure, or specific obligations.
- **📊 Vendor Benchmarking**: Upload multiple contracts to identify the most favorable terms side-by-side.
- **�️ Compliance Audit**: Audit your contracts against industry frameworks to ensure regulatory and corporate alignment.
- **🖼️ OCR Engine**: High-accuracy recognition for scanned physical contracts and legacy document archives.

---

## 🛠️ Tech Stack

- **Core**: Python 3.11, Streamlit
- **Intelligence**: Google Gemini Pro (Reasoning), Legal-BERT (Risk Classification)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **OCR**: EasyOCR + PyMuPDF
- **Infrastructure**: Docker & Hugging Face Spaces

---

## 💻 Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/khushshah103/LegalAI-.git
   cd LegalAI-
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   Set your `GEMINI_API_KEY` in a `.env` file.

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

---

## 🏢 Use Cases

- **📈 Modern Procurement**: Transform months of contract vetting into minutes of automated benchmarking.
- **🤝 Freelance Protection**: Shield yourself from "one-way" indemnity and unfair IP assignments before signing.
- **⚖️ Legal Operation Efficiency**: Scale your legal team's output by automating the "first pass" of risk assessment.
- **🛡️ Corporate Compliance**: Instantly audit massive document libraries against updated industry standards and frameworks.

---
*Developed with ❤️ for the Legal Tech Community.*
