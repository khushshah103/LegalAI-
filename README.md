---
title: LegalAI Analyzer
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# ⚖️ LegalAI Analyzer: Professional Contract Intelligence

[![Live Demo](https://img.shields.io/badge/Live-Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/khushshah103/LegalAI)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**LegalAI Analyzer** is a state-of-the-art contract intelligence platform designed for automated legal review, structured risk assessment, and deep document insights. Unlike generic AI, LegalAI is purpose-built to navigate the complexities of legal language with surgical precision.

---

## 🚀 Why LegalAI instead of ChatGPT?

While generic LLMs are impressive, **LegalAI** offers specialized advantages for legal professionals and procurement teams:

| Feature | LegalAI Analyzer | Generic ChatGPT |
| :--- | :--- | :--- |
| **Legal Specialization** | Uses **Legal-BERT**, a transformer model trained exclusively on legal corpora. | General-purpose training; prone to legal "hallucinations." |
| **Structured Risk Scoring** | Provides quantitative risk metrics with a fine-tuned classification layer. | Provides subjective, conversational summaries. |
| **RAG Precision** | Uses **Vector Search (FAISS)** to ensure answers are strictly grounded in *your* document. | Often draws on internal training data, leading to factual errors. |
| **Scanned Document Support** | Integrated **OCR (EasyOCR)** for processing scanned images and non-selectable PDFs. | Poor native support for "image-based" legal PDFs. |
| **Comparative Analysis** | Designed for **Vendor Comparison** (Multi-doc mode) with unified benchmarking. | Single-document chat focus; difficult to cross-compare. |

---

## � Key Features

- **🔍 Executive Summary**: Get a "Counsel's View" summary of any contract in seconds.
- **⚠️ Risk Detection**: Automatically flag predatory clauses, uncapped liability, and imbalanced indemnity.
- **💬 Contract Chat**: Ask complex questions like *"What is the termination notice period?"* or *"Are there any non-compete clauses?"*
- **📊 Vendor Benchmarking**: Upload multiple contracts to compare terms, liabilities, and pricing side-by-side.
- **📄 Legal Entity Extraction**: Automatically identify sensitive entities, dates, and amounts.
- **🖼️ Optical Character Recognition (OCR)**: Process scanned documents and physical contract photos with ease.

---

## 🛠️ Tech Stack

- **Core**: Python 3.11, Streamlit
- **Intelligence**: Google Gemini Pro (Reasoning), Legal-BERT (Risk Classification)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **OCR**: EasyOCR + PyMuPDF
- **Infrastructure**: Docker & Hugging Face Spaces

---

## � Local Setup

Get started locally in minutes:

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
   Create a `.env` file or set your environment variable:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

---

## 🏢 Use Cases

- **Procurement**: Quickly vet vendor contracts for hidden risks.
- **Freelancers**: Ensure IP rights and payment terms are protected.
- **Legal Teams**: Accelerate the "first pass" of contract review.
- **Compliance**: Audit existing document libraries against corporate standards.

---
*Developed with ❤️ for the Legal Tech Community.*
