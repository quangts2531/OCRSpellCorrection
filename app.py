import os
import logging
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from image_to_text import ImageToText

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

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

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

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
            logger.error("OCR processing failed for %s: %s", filename, e)
            return jsonify({'error': str(e)}), 500
        finally:
            # FIX 4: Always clean up the uploaded file
            try:
                os.remove(filepath)
            except OSError:
                logger.warning("Could not delete temporary file: %s", filepath)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum allowed size is 10 MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, port=7860)
