from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from fastai.vision.all import load_learner, PILImage

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

model_path = os.path.join(BASE_DIR, "mammogram_model.pkl")

import pathlib
import sys

if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner(model_path)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    results = []
    results_single = []

    return render_template("index.html", results=results, results_single=results_single)


@app.route("/classify-mammography", methods=["POST"])
def mammography():
    results = []
    mammograph = []
    uploaded_files = request.files.getlist("single_files")

    for file in uploaded_files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            img = PILImage.create(file_path)
            pred_class, pred_idx, probs = learn.predict(img)
            mammograph.append({
                "filename": filename,
                "predicted_class": str(pred_class),
                "confidence": f"{probs[pred_idx]:.4f}",
                "image_url": f"/uploads/{filename}"
            })

    return render_template("index.html", results=results, results_single=mammograph)
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(debug=True)
