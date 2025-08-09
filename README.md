# Medical Chatbot — Production-ready Demo

**Overview**
A sleek, interview-friendly Medical Chatbot project showcasing NLP, RAG-style retrieval, TensorFlow (optional), and a polished Streamlit frontend. Designed to be uploaded to GitHub and demoed locally or via Docker/Streamlit Cloud.

**What's included**
- `app/streamlit_app.py` — attractive Streamlit UI for chat + file upload + sources & citations panel.
- `backend/build_faq_index.py` — builds an embeddings index from `sample_data/faqs.csv` (uses sentence-transformers + FAISS).
- `backend/chat_backend.py` — simple RAG pipeline: retrieve top-k passages, call a local or API LLM (placeholder), and return response + sources.
- `notebooks/` — notebook walkthrough for data preparation and demo.
- `sample_data/faqs.csv` — small FAQ dataset (sample) used for quick demo.
- `assets/` — logo and hero image placeholders.
- `requirements.txt` — pinned dependencies.
- `Dockerfile` — simple container to run Streamlit app.
- `MODEL_CARD.md`, `PROJECT_SUMMARY.pdf` (placeholder), MIT `LICENSE`.

**Quickstart (local)**
1. Create virtual environment and install packages:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Build FAQ index (creates `indexes/faqs.faiss`):
   ```bash
   python backend/build_faq_index.py --input sample_data/faqs.csv --output indexes/
   ```
3. Run the Streamlit demo:
   ```bash
   streamlit run app/streamlit_app.py
   ```
4. Use the chat UI. The app shows retrieved sources and confidence scores.

**Notes**
- This repo contains a small sample dataset and a RAG pipeline that uses sentence-transformers locally. For production, swap the LLM call in `chat_backend.py` to an API (OpenAI, Anthropic) or a hosted model.
- See `notebooks/` for step-by-step reproduction and model explainability examples.
