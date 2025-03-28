import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Paths
MODEL_PATH = "trained_model"
TOKENIZER_PATH = "trained_model"

# Class label mapping (modify based on your dataset)
class_labels = {
    0: "greetings",
    1: "goodbye",
    2: "self-esteem-A",
    3: "depression",
    4: "self-esteem-B",
    5: "relationship-b",
    6: "relationship-a",
    7: "angermanagement-a",
    8: "angermanagement-b",
    9: "domesticviolence",
    10: "griefandloss",
    11: "substanceabuse-a",
    12: "substanceabuse-b",
    13: "family-conflict",
    # Add more categories if needed
}

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()


def predict(text):
    """Function to get model predictions on a given text"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)  # Apply softmax
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    # Get the corresponding label name
    predicted_label = class_labels.get(predicted_class, "Unknown")
    return predicted_label


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a message (or type 'exit' to stop): ")
        if user_input.lower() == "exit":
            break

        prediction = predict(user_input)
        print(f"Predicted Intent: {prediction}")
