import streamlit as st
import tempfile
import os
from helpers import extract_text_from_pdf, extract_text_from_txt, load_model
from vectorstore import build_vector_store, retrieve_relevant_chunks
from agents import HealthcareAgent, FinancialAgent, GeneralAgent

def main():
    st.title("📝 RAG-based Text Summarization with Multi-Agent Model")
    st.write("""
        Upload a PDF or TXT document, and the system will generate a comprehensive summary
        using Retrieval-Augmented Generation and Multi-Agent Modeling.
    """)

    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if uploaded_file.type == "application/pdf":
            st.write("📄 Extracting text from PDF...")
            text = extract_text_from_pdf(tmp_file_path)
        else:
            st.write("📄 Extracting text from TXT file...")
            text = extract_text_from_txt(tmp_file_path)

        os.unlink(tmp_file_path)

        if text:
            st.write("📝 **Original Text:**")
            st.write(text[:1000] + '...')

            st.write("🔍 Splitting text and building vector store...")
            vector_store, chunks = build_vector_store(text)

            st.write("🤖 Loading summarization model and agents...")
            summarizer = load_model()
            healthcare_agent = HealthcareAgent(summarizer)
            financial_agent = FinancialAgent(summarizer)
            general_agent = GeneralAgent(summarizer)

            st.write("✂️ Generating domain-specific summaries...")
            healthcare_summary = healthcare_agent.process(
                retrieve_relevant_chunks(vector_store, "medical content")
            )
            financial_summary = financial_agent.process(
                retrieve_relevant_chunks(vector_store, "financial content")
            )
            general_summary = general_agent.process(
                retrieve_relevant_chunks(vector_store, "general content")
            )

            st.write("📝 **Generated Summaries:**")
            st.write("### Healthcare Summary")
            st.write(healthcare_summary)

            st.write("### Financial Summary")
            st.write(financial_summary)

            st.write("### General Summary")
            st.write(general_summary)
        else:
            st.warning("No text extracted from the uploaded file.")

if __name__ == "__main__":
    main()
