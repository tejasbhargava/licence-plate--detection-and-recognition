import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import cv2
import easyocr
from ultralytics import YOLO

# -------------------------------
# Load models ONCE
# -------------------------------
model = YOLO("models/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# -------------------------------
# OCR helper
# -------------------------------
def normalize_plate(text):
    text = text.upper()
    text = text.replace("I", "1")   # I → 1
    text = text.replace("O", "0")   # O → 0
    text = text.replace(" ", "")
    return text


def read_plate(image, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    results = reader.readtext(
        gray,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = ""
    confs = []

    for _, t, c in results:
        text += t
        confs.append(c)

    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # 🔥 APPLY NORMALIZATION HERE
    text = normalize_plate(text)

    if 4 <= len(text) <= 12 and confs and sum(confs) / len(confs) > 0.25:
        return text

    return ""

# -------------------------------
# Shared frame processing
# -------------------------------
def process_frame(frame):
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])

        plate_text = read_plate(frame, (x1, y1, x2, y2))
        label = plate_text if plate_text else model.names[cls]

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return frame
# -------------------------------
# Video → Video
# -------------------------------
def video_to_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

# -------------------------------
# Image → Image
# -------------------------------
def image_to_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read image")

    result = process_frame(img)
    cv2.imwrite(output_path, result)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    #video_to_video("test/carLicence1.mp4", "output/output.mp4")
    image_to_image("test/test_img_two.jpg", "output/result_two.jpg")




