from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import onnxruntime as ort
import base64
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ---------------- MODEL ----------------
MODEL_PATH = "yolov8n.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ---------------- UTIL ----------------
def nms(boxes, scores, conf=0.4, iou=0.5):
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    return idx.flatten() if len(idx) else []

# ---------------- SOCKET ----------------
@socketio.on("frame")
def handle_frame(data):
    img_bytes = base64.b64decode(data.split(",")[1])
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # preprocess
    blob = cv2.resize(img, (640, 640))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB) / 255.0
    blob = np.transpose(blob, (2,0,1))[None].astype(np.float32)

    preds = session.run([output_name], {input_name: blob})[0]
    preds = np.squeeze(preds).T

    boxes, scores, classes = [], [], []

    for p in preds:
        cls = np.argmax(p[4:])
        conf = p[4 + cls]
        if conf > 0.4:
            cx, cy, bw, bh = p[:4]
            x = int((cx - bw/2) * w / 640)
            y = int((cy - bh/2) * h / 640)
            bw = int(bw * w / 640)
            bh = int(bh * h / 640)

            boxes.append([x, y, bw, bh])
            scores.append(float(conf))
            classes.append(cls)

    results = []
    for i in nms(boxes, scores):
        x,y,bw,bh = boxes[i]
        results.append({
            "label": CLASSES[classes[i]],
            "box": [x, y, x+bw, y+bh]
        })

    emit("detections", results)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port)
