from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, AutoTokenizer

app = Flask(__name__)

# Load trained model
model_path = "trained_model"  # Path to your trained model
model = GPT2LMHeadModel.from_pretrained(
    model_path, config=model_path, is_decoder=True
)  # Add is_decoder=True
tokenizer = AutoTokenizer.from_pretrained(model_path)


@app.route("/")
def index():
    # Return a simple front-end template for chatting
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Therapist Chat</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #fff3b0;
                    color: #333;
                    text-align: center;
                    padding: 50px;
                }
                h1 {
                    color: #f79c42;
                }
                .chat-container {
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .message-box {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border: 1px solid #ddd;
                }
                .button {
                    background-color: #f79c42;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                .button:hover {
                    background-color: #f57c00;
                }
                .response {
                    background-color: #eaeaea;
                    padding: 10px;
                    margin-top: 10px;
                    border-radius: 5px;
                    max-width: 80%;
                    margin-left: auto;
                    margin-right: auto;
                    text-align: left;
                }
            </style>
        </head>
        <body>
            <h1>Therapist Chat</h1>
            <div class="chat-container">
                <textarea id="userInput" class="message-box" placeholder="Enter your message..." rows="3"></textarea><br>
                <button class="button" onclick="sendMessage()">Send</button>
                <div id="chatOutput"></div>
            </div>
            <script>
                function sendMessage() {
                    const userInput = document.getElementById('userInput').value;
                    if (userInput.trim() === "") return;

                    const chatOutput = document.getElementById('chatOutput');
                    chatOutput.innerHTML += `<div class="response"><strong>You:</strong> ${userInput}</div>`;
                    document.getElementById('userInput').value = '';

                    // Send the message to the Flask backend
                    fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: userInput }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        chatOutput.innerHTML += `<div class="response"><strong>Therapist:</strong> ${data.response}</div>`;
                        chatOutput.scrollTop = chatOutput.scrollHeight;  // Scroll to the bottom
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
                }
            </script>
        </body>
        </html>
    """


@app.route("/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    # Tokenize the user input and generate a response with adjusted parameters
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(
        inputs,
        max_length=150,
        temperature=0.7,  # Control randomness (0.7 is typical, lower for more deterministic results)
        top_p=0.9,  # Control diversity, higher for more diversity
        do_sample=True,  # Enable sampling to generate varied responses
        top_k=50,  # Restrict to top-k options to avoid generating too many random tokens
        num_return_sequences=1,  # Only return one response
    )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
