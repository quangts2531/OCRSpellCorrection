import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from image_to_text import ImageToText

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR engine
ocr_engine = ImageToText()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:

            extracted_text = ocr_engine.image_to_text(filepath)
            
            return jsonify({'text': extracted_text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=7860)
