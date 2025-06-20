# scripts/2_extract_images.py
import fitz
import os
from PIL import Image
import pandas as pd

def extract_images(pdf_path, output_dir='data/images/', doc_title='sample'):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    data = []
    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"{doc_title}_page{i}_{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            data.append({
                'title': doc_title,
                'page': i,
                'image_path': image_path,
                'caption': ''  # Add OCR/caption if needed
            })
    return pd.DataFrame(data)

df_img = extract_images('data/docs/sample.pdf')
df_img.to_csv('data/image_metadata.csv', index=False)
