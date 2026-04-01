import os
# Cần set biến môi trường này trước khi import paddle để tắt MKLDNN/OneDNN trên CPU cho bản 3.x
os.environ['FLAGS_use_mkldnn'] = '0'

import cv2
import textwrap
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# Khởi tạo mô hình PP-OCRv4 (Sử dụng CPU)
# use_angle_cls=True cho bản paddlepaddle 2.x
# show_log=False để tắt log
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, ocr_version='PP-OCRv4', show_log=False)

def extract_text_from_image(image_path_or_array):
    """
    Hàm nhận vào đường dẫn ảnh hoặc ma trận ảnh numpy (từ cv2),
    trả về text hoàn chỉnh từ các box OCR phân tích được.
    """
    print("Đang tiến hành trích xuất chữ bằng PP-OCRv4 (CPU)...")
    
    # Thực hiện inference OCR bằng ocr() cho bản 2.x
    result = ocr_model.ocr(image_path_or_array, cls=True)

    extracted_lines = []
    
    if result is None or len(result) == 0 or result[0] is None:
        print("Không nhận diện được đoạn text nào!")
        return ""

    # Kết quả của thư viện là 1 list, result[0] chứa thông tin cho từng dòng
    # Mỗi line (box_info) có format: [ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ('Text', Confidence) ]
    for box_info in result[0]:
        box_coords = box_info[0]
        text_content = box_info[1][0]
        confidence = box_info[1][1]
        
        # Thiết lập một ngưỡng lọc tự động nhỏ để loại chi tiết rác (ví dụ score < 70%)
        if confidence > 0.7:
            extracted_lines.append(text_content)

    final_text = "\n".join(extracted_lines)
    return final_text

def visualize_ocr(image_path, text_output):
    """
    Vẽ ảnh lên kèm text xuất ra bên cạnh
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh để hiển thị!")
        return

    # Chuyển đổi BGR sang RGB cho Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Cột 1: Ảnh gốc
    axes[0].imshow(img_rgb)
    axes[0].axis('off')
    axes[0].set_title('Ảnh crop đầu vào (Note)', fontsize=14)

    # Cột 2: Text trích xuất được
    axes[1].axis('off')
    axes[1].set_title('Text trích xuất (OCR)', fontsize=14)
    # textwrap giúp wrap từ để không bị tràn màn hình khi hiển thị
    wrapped_text = "\n".join([textwrap.fill(line, width=50) for line in text_output.split('\n')])
    axes[1].text(0.05, 0.95, wrapped_text, transform=axes[1].transAxes, 
                 fontsize=12, verticalalignment='top', 
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    import sys
    
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
    else:
        # Đường dẫn ảnh test mặc định (bạn tự thay thành một đoạn chữ nhỏ lúc chạy test thử)
        test_img_path = "/media/quyet/01DAD374BE175C40/Technical_Draw_Detection/OCR_Text/test_image/image1.png" 

    if os.path.exists(test_img_path):
        detected_text = extract_text_from_image(test_img_path)
        
        print("\n" + "="*40)
        print("=== KẾT QUẢ TEXT MÀ OCR ĐỌC ĐƯỢC ===")
        print("="*40)
        print(detected_text)
        print("="*40 + "\n")
        
        visualize_ocr(test_img_path, detected_text)
    else:
        print(f"Không tìm thấy file: {test_img_path}")
        print("Cách dùng: python ocr_inference.py <đường_dẫn_tới_ảnh_Crop_Note>")