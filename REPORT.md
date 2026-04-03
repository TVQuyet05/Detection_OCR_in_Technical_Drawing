# 📐 Technical Drawing Detection & OCR Project

---

## 1. 📊 Dataset

### 1.1 Dataset chính (Dataset_main)

- Gán nhãn **57 ảnh bản vẽ kỹ thuật** bằng Roboflow
- Chia dữ liệu:
  - **30 ảnh train**
  - **27 ảnh validation**
- Augmentation:
  - Tăng từ 30 → **90 ảnh train**

👉 Kết quả:

- Dataset ban đầu: **90 train / 27 valid**

---

### 1.2 Mở rộng dữ liệu (Data_stage1)

- Thu thập thêm **2 dataset technical drawing trên Roboflow**
- Gộp lại thành:
  - **[Data_stage1](https://drive.google.com/drive/folders/1epL1o3zXr_tbBwyoRoWCPlj2q-opj8BF?usp=sharing)**

---

### 1.3 Kết hợp dữ liệu (Data_stage2)

- Lấy thêm **90 ảnh từ Data_stage1/valid**
- Gộp vào Dataset_main

👉 Dataset sau khi mở rộng:

- **180 train / 27 valid**
- Gọi là: **[Data_stage2](https://drive.google.com/drive/folders/1BDvFjqtoybkZqCuujIoPi-AKFmNyTP2p?usp=sharing)**

---

## 2. 🧠 Model Detection

### 2.1 Phương pháp

- Sử dụng **Faster R-CNN (Detectron2)**
- Fine-tune từ **pretrained model (Model Zoo)**

---

### 2.2 Training Strategy

- Training qua **2 stage**:
  - **Stage 1**: Train trên `Data_stage1` với Pretrained Model (Model Zoo).
  - **Stage 2**: Fine-tune tiếp trên `Data_stage2`, sử dụng trực tiếp checkpoint tốt nhất đạt được sau khi kết thúc Stage 1.

---

### 2.3 Kết quả

- Đánh giá trên Dataset_main/ valid:
  - **mAP@50 tăng từ 0.667 → 0.765**

---

### 2.4 Model đã train

- Checkpoint:
  - `model_final_2.pth`
- Upload tại:
  - https://huggingface.co/TVQuyet05/Detection_Tech_Draw

---

### 2.5 Cải thiện

**Vấn đề:**

- Bounding box bị chồng lấn nhiều

**Giải pháp:**

- Điều chỉnh **NMS (Non-Maximum Suppression)** khi inference

👉 Hiệu quả:

- Giảm bbox trùng
- Tăng chất lượng detection

---

## 3. 🔍 Model OCR - Note

### 3.1 Model sử dụng

- **PP-OCRv4 (PaddleOCR)**

### 3.2 Đặc điểm

- Phù hợp chạy trên CPU
- Độ chính xác cao

---

## 4. 📋 Model OCR - Table

### 4.1 Model sử dụng

- **SLANet (Structure Learning Attention Network)**

👉 Mô tả:

- Model chuyên nhận diện **cấu trúc bảng**
- Trích xuất:
  - Hàng (rows)
  - Cột (columns)
  - Cell

---

### 4.2 Xử lý

- OCR từng cell
- Xuất kết quả dưới dạng **HTML table**

---

### 4.3 Hạn chế

- Không bảo toàn được chính xác kiến trúc bảng với:
  - Bảng phức tạp
  - Bảng nhiều merge cell
  - ảnh chất lượng thấp



## 5. 🔄 Pipeline dữ liệu

### 5.1 Flow tổng thể


- Input Image -> Detection (Faster R-CNN) -> Crop từng vùng (bbox) -> Phân loại theo label (Note → OCR (PP-OCRv4)) (Table → OCR (SLANet)) -> Post-processing -> Output JSON (kết quả detection và OCR)
---

## 6. 🚀 Deploy

### 6.1 Nền tảng

- Deploy trên **Hugging Face Spaces**
- **Demo Website:** [Technical Draw Detection UI](https://huggingface.co/spaces/TVQuyet05/Technical_Draw_Detection)

### 6.2 Công nghệ

- Gradio (UI)
- Hỗ trợ chạy Inference Pipeline (CLI) trước khi khởi tạo UI trên deploy.

---

### 6.3 Đặc điểm

- Chạy trên **CPU (≈12GB RAM)**
- Phù hợp demo pipeline end-to-end:
  - Detection + OCR + Table parsing

---

## 7. 📌 Tổng kết

### Ưu điểm

- Pipeline hoàn chỉnh:
  - Detect → Crop → OCR → Output
- Model detection cải thiện rõ rệt (mAP tăng)
- Có khả năng xử lý cả:
  - Text thường
  - Bảng

---

### Hạn chế

- OCR bảng chưa ổn định
- Một số bbox vẫn cần refine thêm
- Phụ thuộc tài nguyên CPU khi deploy

---

### Hướng phát triển

- Cải thiện Table Structure Recognition
- Thử các model detection mới
- Tối ưu tốc độ inference khi deploy

---
