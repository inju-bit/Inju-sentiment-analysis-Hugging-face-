import streamlit as st
from transformers import pipeline


# Load the Hugging Face AI model
@st.cache_resource
def load_model():
    model = pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    return model


# Create our AI model
classifier = load_model()


# Create the website
st.title("🤗 AI Sentiment Analyzer")

st.write("Enter a sentence to analyse its sentiment.")

# Get text from the user
user_text = st.text_area("Enter your text:")


# Create an Analyze button
if st.button("Analyze Sentiment"):

    if user_text.strip():

        # Send the text to the AI model
        result = classifier(
            user_text,
            truncation=True,
            max_length=512
        )[0]

        # Get the prediction
        label = result["label"]
        confidence = result["score"]

        # Display the result
        if label == "POSITIVE":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

    else:
        st.warning("Please enter some text.")