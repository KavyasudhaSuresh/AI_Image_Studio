from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
from models.styletransfer.fnst import stylize_image
from models.stable_diffusion.textToimage import generate_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styletransfer.html')
def styletransfer_page():
    return render_template('styletransfer.html')

@app.route('/text2image.html')
def text2image_page():
    return render_template('text2image.html')

@app.route('/highres.html')
def highres_page():
    return render_template('highres.html')

@app.route('/about.html')
def about_page():
    return render_template('about.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = f"stylized_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    stylize_image(input_path, output_path)
    return jsonify({'output_image': f'/{output_path}'})

@app.route('/text2image', methods=['POST'])
def text2image():
    try:
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        output_filename = f"generated_{secure_filename(prompt[:10])}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        generate_image(prompt, output_path)

        return jsonify({'url': '/' + output_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
