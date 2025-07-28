import cv2
import numpy as np
import os

def main():
    L = int(input("Nhập số lượng vector trội: ")) #200

    # Kích thước ảnh yêu cầu
    M, N = 100, 100


    # chuyển ảnh màu thành ảnh xám
    image_folder = "training_db"
    image_files = os.listdir(image_folder)
    n = len(image_files)

    # Khởi tạo ma trận 0 n hàng M*N cột ([X] = 2562x1000)
    X = np.zeros((n, M * N)) 
    
    for i, file in enumerate(image_files):
        path = os.path.join(image_folder, file)

        # đọc ảnh xám
        I = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  


        # Tiền xử lý: cân bằng histogram + giảm nhiễu
        I = cv2.equalizeHist(I)
        I = cv2.GaussianBlur(I, (3, 3), 0)

        # resize về M * N
        I = cv2.resize(I, (N, M))                   

        # chuyển ảnh thành vector 1 chiều
        X[i, :] = I.flatten()                       

    # Tính giá trị trung bình của tất cả các ảnh
    m = np.mean(X, axis=0)

    # Chuẩn hóa dữ liệu bằng cách trừ mean của mỗi cột khỏi từng hàng 
    X -= m

    # Tính ma trận hiệp phương sai theo công thức (X^T * X) / (N - 1)
    n_samples = X.shape[0]
    C = (X.T @ X) / (n_samples - 1)

    # Chọn thành phần chính (principle component)
    # Lấy giá trị riêng, vector riêng
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sắp xếp chỉ số thứ tự của giá trị riêng trong ma trận theo hướng giảm dần
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # Tính ma trận biến đổi PCA để giảm chiều (lấy L giá trị riêng ứng với L vector trội)
    PCA = eigenvectors_sorted[:, :L]

    # Chiếu dữ liệu vào không gian PCA (kết quả [T] = 2562 x 100)
    T = X @ PCA

    # lưu không gian biến đổi PCA, giá trị trung bình và không gian sau đặc trưng
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "pca_matrix.npy"), PCA)
    np.save(os.path.join(output_dir, "mean_vector.npy"), m.real)
    np.save(os.path.join(output_dir, "feature_matrix.npy"), T)
    np.save(os.path.join(output_dir, "db_images.npy"), np.array(image_files))

    return
if __name__ == "__main__":
    main()