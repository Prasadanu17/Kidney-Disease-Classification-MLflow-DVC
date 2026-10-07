from flask import Flask, request,jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')

app=Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename="inputImage.jpg"
        self.classifier=PredictionPipeline(self.filename)

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json(silent=True) or {}
        image = data.get('image')
        if not image:
            return jsonify([{"error": "No image data provided"}]), 400
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify([{"error": str(e)}]), 500

@app.route("/sample/<category>", methods=['GET'])
@cross_origin()
def sampleRoute(category):
    try:
        import glob, base64
        folder = "Normal" if "normal" in category.lower() else "Tumor"
        pattern = os.path.join("artifacts", "data_ingestion", "kidney-ct-scan-image", folder, "*.jpg")
        files = glob.glob(pattern)
        if not files:
            files = glob.glob(f"**/{folder}/*.jpg", recursive=True)
        if files:
            with open(files[0], "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return jsonify({"image": b64, "filename": os.path.basename(files[0]), "label": folder})
        return jsonify({"error": f"No {folder} sample found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port)