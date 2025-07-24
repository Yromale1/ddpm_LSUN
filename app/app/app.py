from flask import Flask, request, jsonify, send_from_directory
from generate import generate_image

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    label = request.json.get("label", 0)
    img_base64 = generate_image(label)
    return jsonify({"image": img_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)