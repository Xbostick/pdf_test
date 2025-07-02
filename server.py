"""
Flask server for uploading PDF files and predicting their newsgroup category.
"""
from flask import Flask, request, redirect, render_template
from pathlib import Path
from dl_functions import predict 
from text_helper import load_preprocessor, tokenize_text_inference
from labels_newsground import labels_newsground
app = Flask(__name__)


# Initialize Flask app and configurations
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('./files')
app.config['CONTENT_FOLDER'] = Path('./pages')
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# check if upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Load preprocessor
preprocessor = load_preprocessor()

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads and redirect to prediction page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            # Assert file is present in the request
            return redirect(request.url)
        file = request.files['file']

        # Assert file has a valid name
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(app.config['UPLOAD_FOLDER'] / filename)
            return redirect(f'/complete_{filename}')
        
    return render_template('download_page.html')

@app.route('/complete_<string:path>')
def complete_download(path):
    """
    Process uploaded file and return predicted newsgroup category.
    """
    file_path = app.config['UPLOAD_FOLDER'] / path
    # Assert file exists
    assert file_path.exists(), f"File {path} not found"
    
    text = tokenize_text_inference(file_path, preprocessor)
    result = predict(text, model_type="Linear")
    return f'Completed! {path} is predicted as {labels_newsground[result[0]]} (class {result[0]})'

if __name__ == "__main__":
    app.run(debug=True)