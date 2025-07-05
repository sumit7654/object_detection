import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

## python -m streamlit run detection.py run this for live
# Streamlit page config
st.set_page_config(page_title="Object Detection By Sumit", layout="centered")
st.title("🎯Object Detection App BY SUMIT ")

# Model selection
model_choice = st.selectbox("Select YOLOv8 Model", ["yolov8n.pt", "Upload custom model"])
if model_choice == "Upload custom model":
    uploaded_model = st.file_uploader("Upload your custom .pt model", type=["pt"])
    if uploaded_model:
        model_path = os.path.join("custom_model.pt")
        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())
        model = YOLO(model_path)
    else:
        st.warning("Upload a YOLO model to continue.")
        st.stop()
else:
    model = YOLO(model_choice)

# Optional class filter
all_classes = list(model.names.values())
selected_classes = st.multiselect("Filter by class (optional):", all_classes)

# Input type
input_type = st.radio("Choose Input Type", ["Image", "Video", "Webcam"])

# Inference function
def detect_objects(image_np):
    results = model.predict(image_np, conf=0.3)
    output = results[0].plot()
    
    if selected_classes:
        filtered_img = image_np.copy()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in selected_classes:
                    filtered_img = r.plot()
        return filtered_img
    return output

# --- Image Input ---
if input_type == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        image_np = np.array(image)
        result_img = detect_objects(image_np)
        st.image(result_img, caption="Detected Image", channels="BGR")

# --- Video Input ---
elif input_type == "Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid:
        temp_video_path = tempfile.NamedTemporaryFile(delete=False).name + ".mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_vid.read())

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = detect_objects(frame)
            stframe.image(result_frame, channels="BGR", use_column_width=True)
        cap.release()

# --- Webcam Input using streamlit-webrtc ---
elif input_type == "Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result_img = detect_objects(img)
            return result_img

    webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer
    )
