from flask import Flask, jsonify, request
from flask_cors import CORS
import os

# app instance
app = Flask(__name__)
CORS(app)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/home', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the API'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return jsonify({'message': f'File {file.filename} uploaded successfully!'})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
