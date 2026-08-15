# 📄 Document Processing Pipeline

> A production ready Python document processing pipeline for AI, Retrieval Augmented Generation (RAG) and Large Language Model (LLM) applications.

The **Document Processing Pipeline** is designed to transform raw and unstructured PDF documents into clean, structured and AI ready data. It automates document extraction, OCR, text cleaning, metadata generation, section detection, table processing, validation and export, providing high quality input for modern AI systems.

This project is being developed as the document processing component of **AgriNovaAI**, but it is designed as a standalone and reusable pipeline that can also be used in healthcare, legal, finance, education, research and enterprise AI applications.

---
<br>

# 🚀 Features

* 📄 Automatic PDF processing
* 📂 Recursive folder scanning
* 🔍 OCR support for scanned PDFs
* 🧹 Intelligent text cleaning
* 📑 Automatic metadata extraction
* 📚 Document section detection
* 📊 Table extraction and conversion
* ✅ Document quality validation
* 💾 Export to TXT, Markdown and JSON
* 🤖 AI ready output for RAG and LLM applications

---
<br>

# 🏗 Project Architecture

```text
Raw PDF Documents
        │
        ▼
PDF Reader
        │
        ▼
OCR (Scanned Documents)
        │
        ▼
Text Cleaning
        │
        ▼
Metadata Extraction
        │
        ▼
Section Detection
        │
        ▼
Table Extraction
        │
        ▼
Document Validation
        │
        ▼
Export Clean Documents
        │
        ▼
Ready for RAG
```

---
<br>

# 📂 Project Structure

```text
document-processing-pipeline/
│
├── README.md
├── requirements.txt
├── config.py
├── pdf_reader.py
├── ocr.py
├── cleaner.py
├── metadata.py
├── section_splitter.py
├── table_extractor.py
├── validator.py
├── exporter.py
├── pipeline.py
├── main.py
├── input_pdfs/
└── output/
    ├── clean_text/
    ├── markdown/
    ├── metadata/
    └── reports/

```

---
<br>

# ⚙️ Processing Workflow

The pipeline automatically processes every PDF through the following stages:

1. Read PDF documents
2. Detect scanned pages
3. Extract text using OCR (if required)
4. Clean and normalize text
5. Extract metadata
6. Detect document sections
7. Extract tables
8. Validate extraction quality
9. Export cleaned outputs
10. Prepare documents for RAG

---
<br>

# 📦 Installation

## Clone the Repository

```bash
git clone https://github.com/DewmikaSenarathna/AgriNova-AI/tree/main/Document-Processing-Pipeline

cd Document-Processing-Pipeline
```

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

---
<br>

# 🔍 OCR Requirements

OCR is only required for scanned or image based PDF documents.

Install the required system packages:

### Ubuntu / Debian

```bash
sudo apt install tesseract-ocr poppler-utils
```

### macOS

```bash
brew install tesseract poppler
```

### Windows

* Install **Tesseract OCR**
* Install **Poppler for Windows**
* Add both to your system PATH

If OCR is not installed, the pipeline will continue processing normal text based PDFs.

---
<br>

# 📁 Input Documents

Place all PDF documents inside the `input_pdfs/` directory.

Example:

```text
input_pdfs/
├── Soil/
│   ├── soil_report.pdf
│   └── fertility.pdf
│
├── Crop/
│   ├── rice.pdf
│   └── maize.pdf
│
└── Irrigation/
    └── irrigation_manual.pdf
```

The pipeline automatically scans all folders and subfolders.

---
<br>

# ▶️ Running the Pipeline

Execute the pipeline with a single command:

```bash
python main.py
```

The pipeline will automatically process every available PDF.

---
<br>

# 📤 Output

After processing, the generated files are saved in:

```text
output/
├── clean_text/   ← plain .txt files (one per PDF)
├── markdown/     ← .md files with ## headings per section + tables
├── metadata/     ← .json files: title, language, type, word count, etc.
└── reports/      ← one JSON report per run, showing PASS/FAIL per file
```

---
<br>

# 📋 Pipeline Modules

| File | What it does |
|---|---|
| `config.py` | All settings in one place (folders, thresholds) |
| `pdf_reader.py` | Finds every PDF (even in sub folders) and extracts raw text |
| `ocr.py` | Detects scanned pages and reads text out of the images |
| `cleaner.py` | Removes headers, footers, page numbers, extra spaces |
| `metadata.py` | Guesses title, detects language, guesses document type |
| `section_splitter.py` | Detects headings and splits text into sections |
| `table_extractor.py` | Converts PDF tables into clean Markdown tables |
| `validator.py` | Flags low quality / broken extractions |
| `exporter.py` | Saves the final .txt / .md / .json files |
| `pipeline.py` | Runs all the steps above, in order, for every PDF |
| `main.py` | The one command you actually run |

---
<br>

# 🛠 Technologies

* Python
* PyMuPDF
* PDFPlumber
* Tesseract OCR
* OpenCV
* Pillow
* Camelot
* Pandas
* NumPy
* LangDetect

---
<br>

# 🎯 Current Development Roadmap

* ✅ Project structure
* 🔄 PDF reader
* ⏳ OCR support
* ⏳ Text cleaning
* ⏳ Metadata extraction
* ⏳ Section detection
* ⏳ Table extraction
* ⏳ Validation
* ⏳ Export system
* ⏳ RAG integration

---
<br>

# 🔗 Integration with AgriNovaAI

This project serves as the document processing layer for **AgriNovaAI**.

```text
Raw Agricultural Documents
            │
            ▼
Document Processing Pipeline
            │
            ▼
Chunking
            │
            ▼
Embedding
            │
            ▼
Vector Database
            │
            ▼
Retriever
            │
            ▼
Large Language Model
            │
            ▼
AgriNovaAI Assistant
```

---
<br>
