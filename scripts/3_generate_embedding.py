# scripts/3_generate_embeddings.py
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_texts(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.cpu().numpy()

def embed_images(image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

# Load and embed text
text_df = pd.read_csv('data/text_chunks.csv')
text_df["embedding"] = embed_texts(text_df["text"].tolist()).tolist()

# Load and embed images
img_df = pd.read_csv('data/image_metadata.csv')
img_df["embedding"] = embed_images(img_df["image_path"].tolist()).tolist()

# Save combined
text_df["modality"] = "text"
img_df["modality"] = "image"
combined_df = pd.concat([text_df, img_df], ignore_index=True)
combined_df.to_pickle("embeddings/clip_embeddings.pkl")
