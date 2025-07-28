import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity
import os

# Tải dữ liệu đã được huấn luyện
PCA = np.load("models/pca_matrix.npy")
m = np.load("models/mean_vector.npy")
T = np.load("models/feature_matrix.npy")
db_images = np.load("models/db_images.npy")

# Thư mục chứa database gốc
IMAGE_FOLDER = "training_db"

# Chiếu PCA
def project_image_to_pca(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    vector = img.flatten()
    vector_centered = vector - m
    return vector_centered @ PCA  # (100,)

# Tìm ảnh giống nhất
def find_most_similar(input_feature):
    sims = cosine_similarity([input_feature], T)[0]  # Tính độ tương tự
    idx = np.argmax(sims)  # Chọn ảnh có similarity cao nhất
    return idx, sims[idx]

# Upload và tìm ảnh
def upload_and_search():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        # Hiển thị ảnh input
        input_img = Image.open(file_path).resize((150, 150))
        input_img_tk = ImageTk.PhotoImage(input_img)
        lbl_input.config(image=input_img_tk)
        lbl_input.image = input_img_tk

        # PCA projection + tìm ảnh giống nhất
        feature = project_image_to_pca(file_path)
        idx, dist = find_most_similar(feature)

        # Hiển thị ảnh kết quả
        result_path = os.path.join(IMAGE_FOLDER, db_images[idx])
        result_img = Image.open(result_path).resize((150, 150))
        result_img_tk = ImageTk.PhotoImage(result_img)
        lbl_result.config(image=result_img_tk)
        lbl_result.image = result_img_tk

        lbl_info.config(text=f"Ảnh giống nhất: {db_images[idx]}  |  Khoảng cách: {dist:.2f}")

# UI
root = tk.Tk()
root.title("Image Search with PCA")

frame = tk.Frame(root)
frame.pack(pady=10)

# Nút upload
btn_upload = tk.Button(frame, text="Upload Image", command=upload_and_search)
btn_upload.grid(row=0, column=0, columnspan=2, pady=5)

# Label hiển thị ảnh
lbl_input = tk.Label(frame, text="Input Image")
lbl_input.grid(row=1, column=0, padx=10)

lbl_result = tk.Label(frame, text="Result Image")
lbl_result.grid(row=1, column=1, padx=10)

# Thông tin kết quả
lbl_info = tk.Label(root, text="")
lbl_info.pack(pady=5)

root.mainloop()
