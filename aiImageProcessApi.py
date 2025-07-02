from flask import Flask, request, jsonify
from libs.emptyPageDetect.EmptyPageDetectHelper import trainModel, predict_page
import os

app = Flask(__name__)

@app.route("/api/train", methods=["POST"])
def api_train_model():
    data = request.get_json()
    empty_dir = data.get("empty_dir")
    filled_dir = data.get("filled_dir")

    if not empty_dir or not filled_dir:
        return jsonify({"success": False, "error": "empty_dir ve filled_dir zorunludur"}), 400

    if not os.path.exists(empty_dir) or not os.path.exists(filled_dir):
        return jsonify({"success": False, "error": "Dizin(ler) bulunamadı"}), 404

    try:
        trainModel(empty_dir, filled_dir)
        return jsonify({"success": True, "message": "Model başarıyla eğitildi"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict_page():
    data = request.get_json()
    input_data = data.get("input")
    model_path = data.get("model_path", "../../mlModels/detectEmptyPageModel/page_classifier_enhanced.pkl")

    if not input_data:
        return jsonify({"success": False, "error": "input (path veya base64) zorunludur"}), 400

    try:
        label, pred, prob = predict_page(input_data, model_path=model_path)
        return jsonify({
            "success": True,
            "label": label,
            "prediction": int(pred),
            "probability": round(prob, 2)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

