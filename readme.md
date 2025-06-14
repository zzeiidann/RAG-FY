
# 🔍 RAG-FY: Multimodal RAG System

<div align="center">

![RAG-FY Logo](https://img.shields.io/badge/RAG--FY-Multimodal%20Search-blue?style=for-the-badge&logo=search&logoColor=white)

**A powerful Retrieval-Augmented Generation system that combines image and text embeddings for intelligent multimodal search**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-green.svg)](https://streamlit.io)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-orange.svg)](https://openai.com/research/clip)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-purple.svg)](https://faiss.ai)

[🚀 Quick Start](#-quick-start) • [📚 Features](#-features) • [🛠️ Installation](#️-installation) • [💡 Usage](#-usage) 

</div>

---

## 🎯 What is RAG-FY?

RAG-FY is a cutting-edge **Retrieval-Augmented Generation** system that enables you to:

- 🖼️ **Search by Image**: Upload an image and find similar content
- 🔤 **Search by Text**: Use natural language to find relevant images and descriptions
- 🧠 **Multimodal Understanding**: Leverages OpenAI's CLIP model for joint image-text embeddings
- ⚡ **Fast Vector Search**: Powered by FAISS for lightning-fast similarity search
- 🌐 **Web Interface**: Beautiful Streamlit UI for easy interaction
- 📱 **Command Line**: Terminal-friendly for automation and scripting

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎨 **Multimodal Search**
- Search using images or text queries
- Cross-modal retrieval (text→image, image→text)
- CLIP-based embeddings for semantic understanding

### ⚡ **High Performance**
- FAISS vector database for fast similarity search
- Memory-optimized for large datasets
- Batch processing support

</td>
<td width="50%">

### 🖥️ **Multiple Interfaces**
- Interactive Streamlit web app
- Command-line interface
- Python API for integration

### 🛠️ **Easy Management**
- Simple database creation and management
- Built-in database clearing and reset
- Comprehensive logging and error handling

</td>
</tr>
</table>

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/zzeiidann/RAG-FY.git
cd RAG-FY
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Your Data
Create your knowledge base JSON file:
```json
{
  "image1.jpg": "A beautiful sunset over the mountains",
  "image2.jpg": "A cat sitting on a windowsill",
  "image3.jpg": "Modern architecture with glass windows"
}
```

### 4️⃣ Build the Database
```bash
python Data.py --image_dir ./images --json ./image_knowledge.json
```

### 5️⃣ Start Searching!

**Web Interface:**
```bash
streamlit run main.py
```

**Command Line:**
```bash
# Search by text
python query.py --text "sunset mountains"

# Search by image
python query.py --image "./query_image.jpg"
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (CPU works fine)

### Option 1: Using pip
```bash
# Clone repository
git clone https://github.com/zzeiidann/RAG-FY.git
cd RAG-FY

# Install requirements
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Create environment
conda create -n ragfy python=3.9
conda activate ragfy

# Clone and install
git clone https://github.com/zzeiidann/RAG-FY.git
cd RAG-FY
pip install -r requirements.txt
```

### 📦 Dependencies
```
torch>=1.9.0
transformers>=4.21.0
faiss-cpu>=1.7.2
streamlit>=1.25.0
Pillow>=8.3.0
numpy>=1.21.0
sentence-transformers>=2.2.0
```

## 💡 Usage

### 📁 Project Structure
```
RAG-FY/
├── 🗂️ embedding_function.py    # CLIP embedders (Image & Text)
├── 🗂️ Data.py                  # Database creation script
├── 🗂️ query.py                 # Search functionality
├── 🗂️ main.py                  # Streamlit web interface
├── 📁 images/                   # Your image directory
├── 📄 image_knowledge.json      # Knowledge base mapping
└── 📁 faiss_db/                # Generated database files
```

### 🔧 Core Components

#### 1. **Database Creation** (`Data.py`)
```bash
# Create new database
python Data.py --image_dir ./images --json ./knowledge.json

# Reset existing database
python Data.py --reset --image_dir ./images --json ./knowledge.json
```

**Options:**
- `--image_dir`: Directory containing your images
- `--json`: JSON file mapping images to descriptions
- `--reset`: Clear existing database before creating new one

#### 2. **Search Interface** (`query.py`)
```bash
# Text search
python query.py --text "your search query" --top_k 10

# Image search
python query.py --image "./path/to/image.jpg" --top_k 5
```

**Options:**
- `--text`: Search using text query
- `--image`: Search using image file
- `--top_k`: Number of results to return (default: 5)

#### 3. **Web Interface** (`main.py`)
```bash
streamlit run main.py
```
Opens a beautiful web interface at `http://localhost:8501`

### 📋 Knowledge Base Format

Create a JSON file mapping your images to descriptions:

```json
{
  "beach_sunset.jpg": "A beautiful sunset over a calm beach with golden sand",
  "city_night.jpg": "Urban cityscape at night with illuminated skyscrapers",
  "forest_path.jpg": "A winding path through a dense green forest",
  "cat_portrait.jpg": "Close-up portrait of a fluffy orange tabby cat"
}
```

**Tips for better search results:**
- ✅ Use descriptive, detailed descriptions
- ✅ Include relevant keywords and context
- ✅ Mention colors, objects, scenes, and emotions
- ❌ Avoid very short or generic descriptions

## 🎨 Web Interface Features

<div align="center">

### 🖥️ **Streamlit Dashboard**

</div>

The web interface provides:

- **🔤 Text Search Tab**: Enter natural language queries
- **🖼️ Image Search Tab**: Upload images for similarity search
- **⚙️ Configuration Sidebar**: Adjust search parameters
- **📊 Database Status**: Check system health
- **🎨 Beautiful Results**: Formatted search results with rankings
- **🔄 Memory Management**: Built-in cache clearing

**Interface Features:**
- Drag & drop image uploads
- Real-time search results
- Distance scoring for relevance
- Responsive design for mobile/desktop
- Error handling with user-friendly messages

## 🙏 Acknowledgments

- **OpenAI** for the CLIP model
- **Facebook AI** for FAISS vector search
- **Hugging Face** for the transformers library
- **Streamlit** for the amazing web framework

