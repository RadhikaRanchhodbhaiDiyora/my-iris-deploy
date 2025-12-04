from flask import Flask, request, jsonify, render_template_string
import joblib
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join("model", "iris_model.joblib")
model = joblib.load(MODEL_PATH)
target_names = ["setosa", "versicolor", "virginica"]

INDEX_HTML = """
<!doctype html>
<title>Iris Predictor</title>
<h2>Iris Predictor</h2>
<form method="post" action="/predict_web">
  Sepal length: <input name="sl" value="5.1"><br>
  Sepal width:  <input name="sw" value="3.5"><br>
  Petal length: <input name="pl" value="1.4"><br>
  Petal width:  <input name="pw" value="0.2"><br>
  <button type="submit">Predict</button>
</form>
<div>{{result}}</div>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, result="")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    try:
        features = [
            float(request.form["sl"]),
            float(request.form["sw"]),
            float(request.form["pl"]),
            float(request.form["pw"]),
        ]
        pred = model.predict([features])[0]
        return render_template_string(INDEX_HTML, result=f"Predicted: {target_names[pred]}")
    except Exception as e:
        return render_template_string(INDEX_HTML, result="Error: " + str(e))

@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(force=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Send JSON like {\"features\": [4 numbers]}"})
    features = payload["features"]
    if len(features) != 4:
        return jsonify({"error": "features must be length 4"}), 400
    pred = model.predict([features])[0]
    return jsonify({"prediction": target_names[pred], "class_index": int(pred)})