from transformers import pipeline

# Load a pretrained sentiment analysis model
classifier = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Give the model a sentence to analyse
text = "I love learning artificial intelligence!"

# Ask the model to make a prediction
result = classifier(text)

# Display the prediction
print(result)