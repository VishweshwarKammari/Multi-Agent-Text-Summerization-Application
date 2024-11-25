# **RAG-Based Multi-Agent Text Summarization**

### **📄 Project Overview**
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline combined with a **Multi-Agent System** to generate concise, domain-specific summaries from uploaded documents. It supports **PDF** and **TXT** file uploads and processes them using state-of-the-art NLP models.

### **✨ Features**
1. **Retrieval-Augmented Generation (RAG):**
   - Uses a FAISS-based vector store to retrieve relevant text chunks for summarization.
   - Dynamically focuses on the most contextually relevant parts of the document.

2. **Multi-Agent Model:**
   - Includes domain-specific agents for:
     - Healthcare
     - Financial
     - General summaries
   - Each agent uses specialized prompts for domain-specific text summarization.

3. **Streamlit-Based UI:**
   - User-friendly interface for uploading documents and viewing generated summaries.
   - Dynamic configuration of summary length (min/max).
   - Downloadable summaries for offline use.

4. **Advanced NLP Models:**
   - Leverages `sshleifer/distilbart-cnn-12-6` for summarization.
   - Uses `sentence-transformers/all-MiniLM-L6-v2` for text embeddings in the RAG pipeline.

5. **Document Processing:**
   - Supports text extraction from PDFs (using `pdfminer.six`) and plain text files.

---

### **📂 Repository Structure**
```plaintext
├── app.py                      # Main Streamlit application code
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── utils/
    ├── text_processing.py      # Helper functions for text extraction and processing
    ├── agents.py               # Implementation of domain-specific agents
    ├── vector_store.py         # FAISS-based vector store utilities
    └── summarization.py        # Summarization model setup and execution
