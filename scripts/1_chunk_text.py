# scripts/1_chunk_texts.py
import fitz  # PyMuPDF
import pandas as pd

def chunk_pdf(pdf_path, doc_title):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        chunks.append({
            'title': doc_title,
            'page': page_num,
            'text': text
        })
    return chunks

# Save to CSV
chunks = chunk_pdf('data/docs/sample.pdf', 'sample')
df = pd.DataFrame(chunks)
df.to_csv('data/text_chunks.csv', index=False)
