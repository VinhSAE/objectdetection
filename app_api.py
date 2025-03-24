import os
from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
MODEL_PATH = "models/yolo11n.pt"


def ensure_directory_exists(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📂 Thư mục '{directory}' đã được tạo.")
    else:
        print(f"✅ Thư mục '{directory}' đã tồn tại.")


# Đảm bảo thư mục tồn tại trước khi sử dụng
ensure_directory_exists(UPLOAD_FOLDER)
ensure_directory_exists(OUTPUT_FOLDER)

# Load YOLO model
model = YOLO(MODEL_PATH)

@app.route("/detect", methods=["POST"])
def detect_objects():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Chạy YOLO nhận diện đối tượng
    results = model(filepath)

    # Lưu kết quả vào thư mục output
    output_filename = f"detected_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Kiểm tra nếu model hỗ trợ lưu ảnh
    if hasattr(results[0], "save"):
        results[0].save(output_path)
    else:
        return jsonify({"error": "Model does not support saving results"}), 500

    # Tạo đường dẫn URL chính xác
    download_url = url_for("get_detected_image", filename=output_filename, _external=True)

    return jsonify({
        "message": "Detection completed",
        "download_url": download_url
    })

@app.route("/output/<filename>")
def get_detected_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# Thêm route trang chủ để tránh lỗi 404 khi vào /
@app.route("/")
def home():
    return "YOLO Object Detection API is running!"

# Chạy Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
