# scripts/4_search.py
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_query_embedding(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb.cpu().numpy()

def search(query, k=5):
    df = pd.read_pickle("embeddings/clip_embeddings.pkl")
    query_emb = get_query_embedding(query)
    all_embs = np.vstack(df["embedding"].values)
    sims = cosine_similarity(query_emb, all_embs)[0]
    df["similarity"] = sims
    return df.sort_values("similarity", ascending=False).head(k)

results = search("a car driving through mountains")
print(results[["modality", "page", "text", "image_path", "similarity"]])
