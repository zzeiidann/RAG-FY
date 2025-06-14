# main.py

import streamlit as st
import tempfile
import os
import gc
import torch
from pathlib import Path
from PIL import Image
import logging

# Import the fixed query functions
try:
    from query import search_text, search_image
except ImportError as e:
    st.error(f"Error importing query functions: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Image+Text RAG Search", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.result-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    background-color: #f9f9f9;
}
.result-rank {
    font-size: 1.2rem;
    font-weight: bold;
    color: #e74c3c;
}
.result-filename {
    font-size: 1.1rem;
    font-weight: bold;
    color: #2c3e50;
}
.result-description {
    color: #555;
    margin-top: 0.5rem;
}
.distance-score {
    color: #27ae60;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Image + Knowledge RAG Search</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Search Configuration")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    st.header("📊 System Info")
    db_path = Path("faiss_db")
    if db_path.exists():
        st.success("✅ Database found")
        index_file = db_path / "vector_index.faiss"
        meta_file = db_path / "metadata.pkl"
        if index_file.exists() and meta_file.exists():
            st.info(f"📁 Database path: {db_path}")
        else:
            st.error("Database files incomplete")
    else:
        st.error("Database not found. Run Data.py first!")


col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔤 Text Search")
    query = st.text_input("Enter your search query:", placeholder="e.g., 'red car in the street'")
    
    if st.button("Search with Text", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query")
        else:
            try:
                with st.spinner("Searching..."):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    results = search_text(query, top_k)
                    
                    st.markdown("### 📋 Search Results")
                    if results:
                        for r in results:
                            meta = r['metadata']
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-rank">#{r['rank']}</div>
                                <div class="result-filename">📁 {meta['filename']}</div>
                                <div class="distance-score">Distance: {r['distance']:.4f}</div>
                                <div class="result-description">{meta['description']}</div>
                                <small>Type: {meta['type']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No results found")
                        
            except Exception as e:
                st.error(f"Error during text search: {str(e)}")
                logger.error(f"Text search error: {e}")

with col2:
    st.header("🖼️ Image Search")
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload an image to search for similar content"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Search with Image", use_container_width=True):
            try:
                with st.spinner("Processing image and searching..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image.save(tmp_file.name, 'JPEG')
                        temp_path = tmp_file.name
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    results = search_image(temp_path, top_k)
            
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
           
                    st.markdown("### Search Results")
                    if results:
                        for r in results:
                            meta = r['metadata']
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-rank">#{r['rank']}</div>
                                <div class="result-filename"> {meta['filename']}</div>
                                <div class="distance-score">Distance: {r['distance']:.4f}</div>
                                <div class="result-description">{meta['description']}</div>
                                <small>Type: {meta['type']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No results found")
                        
            except Exception as e:
                st.error(f"Error during image search: {str(e)}")
                logger.error(f"Image search error: {e}")
            finally:
                try:
                    if 'temp_path' in locals():
                        os.unlink(temp_path)
                except:
                    pass


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    💡 Tip: Use descriptive text queries or upload images similar to what you're looking for
</div>
""", unsafe_allow_html=True)

if st.button("🔄 Clear Cache & Refresh"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    st.rerun()