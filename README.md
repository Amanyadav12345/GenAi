# 🔍 Multimodal Search with CLIP
A Python project that enables semantic search across both text and images using OpenAI's CLIP model. Users can input a natural language query and retrieve the most relevant **text chunks** and **images** from a set of documents.

## 🚀 Features
- ✅ Extracts text and images from PDF documents
- ✅ Generates embeddings using [CLIP (ViT-L/14)](https://huggingface.co/openai/clip-vit-large-patch14)
- ✅ Performs cosine similarity-based search over both modalities
- ✅ Streamlit UI for live querying and filtering
- ✅ Extendable with OCR, FAISS, captions, filters, etc.

## 🗂️ Project Structure
See folder layout inside this repo.

## 📦 Setup Instructions
```bash
pip install -r requirements.txt
```

## ⚙️ Pipeline Steps
```bash
python scripts/1_chunk_texts.py
python scripts/2_extract_images.py
python scripts/3_generate_embeddings.py
python scripts/4_search.py
```

## 🖥️ Streamlit UI (Optional)
```bash
streamlit run app/streamlit_app.py
```

## 📌 Future Enhancements
- OCR for images
- FAISS for fast search
- Filter by metadata
- Image-to-text search

## 🤝 License
MIT License
