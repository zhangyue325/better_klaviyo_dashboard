from google import genai
import streamlit as st
import pandas as pd


GEMINI_API_KEY = st.secrets["gemini"]['GEMINI_API_KEY']
DEFAULT_MODEL = "gemini-2.5-flash"

def _get_client():
    api_key = st.secrets["gemini"]['GEMINI_API_KEY']
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .streamlit/secrets.toml")
    return genai.Client(api_key=api_key)

def ask_gemini(prompt: str) -> str:
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
        )
        return response.text or "No response."
    except Exception as e:
        return f"AI error: {e}"
