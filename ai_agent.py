from google import genai
from google.genai import types
import streamlit as st
import pandas as pd
import os


GEMINI_API_KEY = st.secrets["gemini"]['GEMINI_API_KEY']
DEFAULT_MODEL = "gemini-3-flash-preview"

def _get_client():
    api_key = st.secrets["gemini"]['GEMINI_API_KEY']
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .streamlit/secrets.toml")
    return genai.Client(api_key=api_key)

def ask_gemini(prompt: str, file_path) -> str:    
    try:
        client = _get_client()
        
        uploaded_file = client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(display_name=os.path.basename(file_path), mime_type="text/csv")
        )
            
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[
                uploaded_file,
                prompt
            ]
        )
        return response.text or "No response."
    except Exception as e:
        return f"AI error: {e}"
