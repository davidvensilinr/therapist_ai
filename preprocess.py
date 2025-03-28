import json
import re
import pandas as pd


def clean_text(text):
    """Function to clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!?']", " ", text)  # Remove special characters
    return text.strip()


def preprocess_data(input_file, output_file):
    """Load JSON, clean data, and save as CSV."""
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Assuming JSON structure has 'intents' list
    conversations = []
    for entry in data["intents"]:
        tag = entry["tag"]
        for pattern in entry["patterns"]:
            cleaned_pattern = clean_text(pattern)
            # Assign the response randomly or use multiple responses
            cleaned_response = clean_text(entry["resonses"][0])  # Adjust as needed
            conversations.append(
                {"input": cleaned_pattern, "response": cleaned_response, "tag": tag}
            )

    # Convert to DataFrame
    df = pd.DataFrame(conversations)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")


# Run preprocessing
preprocess_data("data/conversations.json", "data/preprocessed.csv")
