# app/streamlit_app.py
import streamlit as st
from scripts import search  # reuse search logic

st.title("Multimodal Search with CLIP")

query = st.text_input("Enter search query:")
if query:
    results = search(query)
    for _, row in results.iterrows():
        st.write(f"**Type**: {row['modality']}, **Page**: {row['page']}, **Score**: {row['similarity']:.2f}")
        if row['modality'] == 'text':
            st.text(row['text'])
        else:
            st.image(row['image_path'], width=300)
