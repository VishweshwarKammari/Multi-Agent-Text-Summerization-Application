import torch
from transformers import pipeline
from pdfminer.high_level import extract_text
import streamlit as st

def load_model():
    """
    Load the summarization pipeline with a BART model.
    """
    model_name = "sshleifer/distilbart-cnn-12-6"
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model_name, device=device)
    return summarizer


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfminer.six.
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_txt(txt_path):
    """
    Extract text from a TXT file.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""
