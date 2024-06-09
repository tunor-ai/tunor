from flask import Flask, request, jsonify
from src.train_bash import train_model  # Assuming you have a train_model function in train_bash.py

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fine-Tuning Service!"

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    model_name = data.get('model_name')
    dataset = data.get('dataset')
    # Add other parameters as needed
    train_model(model_name, dataset)
    return jsonify({"status": "Training started"}), 200

if __name__ == '__main__':
    app.run(debug=True)
