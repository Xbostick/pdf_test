from flask import Flask
from pathlib import Path
from flask import Flask, flash, request, redirect, url_for, render_template

app = Flask(__name__)



UPLOAD_FOLDER = Path('./files')
CONTENT_FOLDER = Path('./pages')

ALLOWED_EXTENSIONS = {
    'pdf',
    }

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(app.config['UPLOAD_FOLDER'] / filename)
            return redirect('/complete')
        
    return render_template('download_page.html')

@app.route('/complete')
def complete_download():
    #TODO user return 
    return f'Completed!'

app.run(debug=True)