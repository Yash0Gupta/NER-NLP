import spacy
import streamlit as st

# Load SpaCy model
@st.cache_resource
def load_model():
    # Download the model and load it
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    return nlp

# Initialize the model
nlp = load_model()

# Streamlit app layout
st.title("NER Data Extraction")
st.write("This app extracts named entities from the input text.")

# Text input from user
text = st.text_area("Enter the text for NER analysis")

# Process the text when the button is pressed
if st.button("Analyze"):
    if text:
        # Process the text using the SpaCy model
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Display the extracted named entities
        if entities:
            st.write("Named Entities:")
            for entity in entities:
                st.write(f"{entity[0]} ({entity[1]})")
        else:
            st.write("No named entities found.")
    else:
        st.write("Please enter some text to analyze.")
