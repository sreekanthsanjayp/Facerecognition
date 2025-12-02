#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os

folder = "dataset/my_face"   # <-- your folder containing 104 images

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []

for file in os.listdir(folder):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(folder, file)
    img = cv2.imread(path)

    if img is None:
        print(f"Cannot read: {file}")
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"No face detected in: {file}")
        continue

    emb = faces[0].embedding
    embeddings.append(emb)

if len(embeddings) == 0:
    print("❌ No embeddings found — check your images.")
else:
    my_embedding = np.mean(np.array(embeddings), axis=0)
    np.save("my_face_embedding.npy", my_embedding)
    print("✅ Saved my_face_embedding.npy")
    print(f"📌 Total valid face samples: {len(embeddings)}")

