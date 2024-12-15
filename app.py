import spacy
import streamlit as st
import pandas as pd
import re
from io import StringIO
from spacy import displacy
import fitz  # PyMuPDF for PDF extraction
from transformers import pipeline

# Load SpaCy model
@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_sm")
    return nlp

# Load summarization model
@st.cache_resource
def load_summarizer():
    summarizer = pipeline("summarization")
    return summarizer

# Initialize models
nlp = load_model()
summarizer = load_summarizer()

# Function to extract text from PDF
@st.cache_data
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to convert dataframe to CSV
def convert_df_to_csv(df):
    csv = df.to_csv(index=False)
    return StringIO(csv).getvalue()

# Streamlit app layout
def app():
    st.title("TEXTLENS")
    st.write("This app performs Named Entity Recognition (NER) and text summarization.")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    text = ""
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted text from the PDF:")
        st.text_area("PDF Text", text, height=300)

    # Manual text input
    else:
        text = st.text_area("Enter the text for analysis")

    # Perform operations when the Analyze button is clicked
    if st.button("Analyze"):
        if text:
            # Summarization
            with st.spinner("Generating summary..."):
                summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("Text Summary")
            st.write(summary)

            # NER Analysis
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Entity Categorization
            entity_types = sorted(set([ent[1] for ent in entities]))
            selected_entity_types = st.multiselect(
                "Select entity types to extract", options=entity_types, default=entity_types
            )
            filtered_entities = [ent for ent in entities if ent[1] in selected_entity_types]

            # Display the extracted entities
            if filtered_entities:
                st.subheader("Named Entities")
                df = pd.DataFrame(filtered_entities, columns=["Entity", "Label"])
                st.dataframe(df)

                # Download button for entities
                csv = convert_df_to_csv(df)
                st.download_button("Download Entities as CSV", csv, "entities.csv", "text/csv")
            else:
                st.write("No named entities found for the selected categories.")
        else:
            st.write("Please enter some text or upload a PDF for analysis.")

# Run the Streamlit app
if __name__ == "__main__":
    app()
