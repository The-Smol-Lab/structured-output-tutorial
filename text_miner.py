import streamlit as st
import requests
import instructor
from openai import OpenAI
from pydantic import create_model, BaseModel, ValidationError, Field
import json
import PyPDF2
from docx import Document
from io import BytesIO
import re

# App configuration
st.set_page_config(
    page_title="TextMinerAI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    .stButton button {background-color: #4CAF50; color: white;}
    .stTextArea textarea {border-radius: 5px;}
    .stSelectbox select {border-radius: 5px;}
    .advanced-field {border-left: 3px solid #4CAF50; padding-left: 10px; margin-top: 5px;}
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("🔍 TextMinerAI")
st.markdown("Extract structured data from unstructured text using AI models")

def fetch_models():
    """Fetch available models from LM Studio"""
    try:
        response = requests.get("http://localhost:1234/v1/models")
        return [model['id'] for model in response.json()['data']]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return []

def extract_text_from_file(file):
    """Extract text from uploaded file (PDF or DOC)"""
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages])
        elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                          "application/msword"]:
            doc = Document(BytesIO(file.getvalue()))
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Sidebar configuration
with st.sidebar:
    st.header("Model Settings")
    models = fetch_models()
    selected_model = st.selectbox("Choose Model", models, index=0 if models else None)
    
    st.header("Data Schema")
    st.markdown("Define up to 5 fields to extract:")
    
    fields = []
    for i in range(5):
        cols = st.columns([3, 3, 1])
        with cols[0]:
            field_name = st.text_input(f"Field {i+1} Name", key=f"name_{i}")
        with cols[1]:
            field_type = st.selectbox(f"Field {i+1} Type", ["str", "int", "float", "bool"], 
                                    key=f"type_{i}")
        with cols[2]:
            is_advanced = st.checkbox("✨", key=f"adv_{i}",
                                    help="Mark as advanced field with description")
        
        field_desc = None
        if is_advanced:
            with st.container(border=True):
                field_desc = st.text_input(f"Description for Field {i+1}", 
                                         key=f"desc_{i}",
                                         placeholder="Enter field description")
        
        if field_name:
            fields.append((field_name, field_type, field_desc))

# Main content area
st.subheader("Input Text")
input_text = st.text_area("Paste your text here", height=200)

st.subheader("Or Upload Document")
uploaded_file = st.file_uploader("Choose PDF/DOC file", type=["pdf", "doc", "docx"])
if uploaded_file:
    extracted_text = extract_text_from_file(uploaded_file)
    if extracted_text:
        input_text += "\n" + extracted_text if input_text else extracted_text

# Processing button
if st.button("Extract Data", use_container_width=True):
    if not fields or not input_text:
        st.warning("Please define data structure and provide input text")
        st.stop()

    # Create fields dictionary
    fields_dict = {}
    for field_name, field_type, field_desc in fields:
        try:
            if field_type == "str":
                type_class = str
            elif field_type == "int":
                type_class = int
            elif field_type == "float":
                type_class = float
            elif field_type == "bool":
                type_class = bool
            else:
                raise ValueError(f"Invalid type '{field_type}'")
            
            if field_desc:
                fields_dict[field_name] = (type_class, Field(description=field_desc))
            else:
                fields_dict[field_name] = (type_class, ...)
        except Exception as e:
            st.error(f"Error with field '{field_name}': {e}")
            st.stop()

    # Dynamically create Pydantic model
    DynamicModel = create_model('DynamicModel', **fields_dict, __base__=BaseModel)

    # Initialize OpenAI client
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )
    )

    # Process extraction
    try:
        with st.spinner("Extracting data..."):
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "Extract the requested information from the text. Include all specified fields."},
                    {"role": "user", "content": input_text},
                ],
                response_model=DynamicModel,
            )
            
            st.subheader("Extracted Data")
            st.json(json.loads(response.model_dump_json()))
            
    except ValidationError as e:
        st.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Example expander
with st.expander("Example Usage"):
    st.markdown("""
    **1. Define Data Schema in Sidebar:**
    - Field 1: `title` (str)
    - Field 2: `author` (str)
    - Field 3: `publication_year` (int)
    - Field 4: `is_bestseller` (bool) - Mark as advanced with description "Whether the book is a bestseller"
    - Field 5: `genre` (str)
    
    **2. Input Text:**
    ```text
    '1984' by George Orwell, published in 1949, remains a classic dystopian novel.
    It has been on the bestseller list for over 300 weeks due to its enduring relevance.
    ```
    
    **3. Output:**
    ```json
    {
        "title": "1984",
        "author": "George Orwell",
        "publication_year": 1949,
        "is_bestseller": true,
        "genre": "dystopian novel"
    }
    ```
    """)