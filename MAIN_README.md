# Technical Draw Detection Pipeline

**Demo Trực tuyến:** [HuggingFace - Technical Draw Detection](https://huggingface.co/spaces/TVQuyet05/Technical_Draw_Detection)


Dự án này là một hệ thống End-to-End tự động trích xuất thông tin từ Bản vẽ Kỹ thuật. Nó ứng dụng **Detectron2** để phân tách bố cục (PartDrawing, Note, Table), kết hợp với sức mạnh của **PaddleOCR (PP-OCRv4 + PPStructure)** để trích xuất Text và cấu trúc Bảng (HTML). Cuối cùng kết xuất ra file JSON kèm UI trực quan trên Gradio.

## 1. Cài đặt Môi trường (Conda / Local)

Dự án được tối ưu hóa và đảm bảo hoạt động ổn định trên Python 3.10. Bạn làm theo trình tự gốc này để tránh xung đột module giữa Torch, Detectron2 và Paddle.

```bash
# Tạo môi trường ảo với Python 3.10
conda create -n tech_draw python=3.10 -y
conda activate tech_draw

# 1. Cài đặt các thư viện lõi và Engine ML (đặc biệt là Torch & Paddle)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle==2.6.2

# 2. Cài đặt thư viện Detectron2 (Compile Runtime)
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"

# 3. Cài đặt toàn bộ Required Packages dự án (Gradio, OpenCV, HuggingFace Hub, v.v...)
pip install -r requirements.txt
```

_(**Lưu ý HuggingFace Spaces Deployment:** File `app.py` đã được hack logic runtime subprocess để tự động build Detectron2 mà không cần cài thủ công bên trong Container. Bạn chỉ cần commit các file cùng `requirements.txt` và `README.md` config base `python_version: "3.10"`)._

## 2. Huấn luyện (Training) Mô hình Detectron2

Hệ thống train theo kiến trúc chia Giai đoạn (Stage 1 & Stage 2) và sử dụng COCO Dataset Format:

### Bước 2.1: Chuẩn bị Dataset

Các file trong `Preprocess_datasets/` dùng để tạo ra các bộ dataset cho 2 stage.

- **Dữ liệu Data_Stage1:** [Tải tại đây](https://drive.google.com/drive/folders/1epL1o3zXr_tbBwyoRoWCPlj2q-opj8BF?usp=sharing)
- **Dữ liệu Data_Stage2:** [Tải tại đây](https://drive.google.com/drive/folders/1BDvFjqtoybkZqCuujIoPi-AKFmNyTP2p?usp=sharing)


### Bước 2.2: Tiến hành Training (GPU bắt buộc)

Bạn cần chạy file train tuỳ theo Giai đoạn. Mã nguồn này sử dụng cấu trúc Mạng **Faster R-CNN (ResNet-50 FPN)**:

```bash
# Train Stage 1 (Model Zoo qua dữ liệu Data_stage1)
python Detection/train.py

# Hoặc Fine-Tuning Stage 2 (Load checkpoint weights của Stage 1 để học tiếp trên Data_stage2)
python Detection/train_stage2.py
```

> Trong đó, thuật toán huấn luyện Giai đoạn 2 (Stage 2) sẽ lấy trực tiếp Checkpoint (model_final.pth) tốt nhất đạt được sau khi kết thúc Stage 1, kết hợp với các dữ liệu đặc thù tăng cường ở Data_stage2 để tiếp tục tinh chỉnh (Fine-tune). Trọng số cuối cùng sẽ tự động ghi ra thư mục `output_model_stage2`.

**Trọng số Model Detection (Model Weights):** [Detection_Tech_Draw (model_final_2.pth)](https://huggingface.co/TVQuyet05/Detection_Tech_Draw)

### Bước 2.3: Đánh giá Model (Evaluation)

Để đánh giá chất lượng mAP, mAR trên tập valid của Data_stage2:

```bash
python Detection/evaluate.py
```

## 3. Khởi chạy Hệ thống Inference & Demo UI

Để nhận diện, bạn có thể tải Model [model_final_2.pth](https://huggingface.co/TVQuyet05/Detection_Tech_Draw) tốt nhất về máy tính.

Trong trường hợp phục vụ chạy local, File này sẽ tự động:

1. Tải Model Weight phát hiện đối tượng tối ưu nhất từ Hugging Face (model_final_2.pth).
2. Nạp Detectron2, PP-OCR, PPStructure-Table lên **RAM.
3. Bật Web Service Server:

```bash
python app.py
```

_(Upload ảnh kĩ thuật -> Click Xử lý -> Tab Json/HTML tự động generate theo thời gian thực)._
