import streamlit as st
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Face Verification", layout="wide")
st.title("🔍 Face Verification (ME vs NOT-ME)")

# Load your precomputed embedding
@st.cache_resource
def load_my_embedding():
    return np.load("my_face_embedding.npy")

# Load InsightFace model
@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

my_embedding = load_my_embedding()
app = load_model()

# Compare two embeddings
def compare_faces(emb1, emb2, thresh=0.38):
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return sim, sim > thresh

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    faces = app.get(img)

    if len(faces) == 0:
        st.warning("⚠ No face detected!")
    else:
        for face in faces:
            emb = face.embedding
            sim, is_me = compare_faces(my_embedding, emb)

            # Draw detection box
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if is_me else (0, 0, 255)
            label = f"ME ({sim:.2f})" if is_me else f"NOT ME ({sim:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Result", use_column_width=True)
