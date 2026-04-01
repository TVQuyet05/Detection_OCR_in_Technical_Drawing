import os
import cv2
from paddleocr import PPStructure

# Khởi tạo mô hình PP-Structure (mặc định dùng SLANet cho nhận dạng cấu trúc bảng)
# Chế độ nhận dạng bảng sẽ lấy ảnh bảng đầu vào, nhận dạng cấu trúc + OCR nội dung -> xuất ra định dạng HTML hoặc Excel
# recovery=True: Gộp kết quả cấu trúc và OCR để tạo ra file excel/HTML hoàn chỉnh
table_engine = PPStructure(layout=False, table=True, ocr=True, show_log=True, recovery=True, lang='en', device='cpu')

def extract_table_from_image(image_path, output_dir="output_test_table"):
    """
    Hàm nhận vào ảnh của một bảng (crop từ Table Detection)
    và xuất ra cấu trúc bảng (thường là HTML) cùng file Excel.
    """
    print(f"Đang phân tích cấu trúc bảng bằng SLANet (CPU) từ ảnh: {image_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh!")
        return

    # Thực hiện dự đoán (sẽ tự động dùng SLANet)
    result = table_engine(img)

    html_result = ""
    # Thông thường bản vẽ trích thẳng qua PPStructure (khi tắt layout) sẽ trả list 1 phần tử là table
    for region in result:
        if region['type'] == 'table':
            # Nội dung HTML phản ánh cấu trúc mảng và nội dung của table
            html_result = region['res']['html']
            print("\n[THÀNH CÔNG] Đã nhận diện cấu trúc bảng!")

            # Lưu HTML ra file để dễ xem
            html_path = os.path.join(output_dir, 'table_result.html')
            with open(html_path, 'w', encoding='utf-8') as f:
               f.write(html_result)
            print(f"-> Đã xuất tệp HTML cấu trúc hiển thị bảng: {html_path}")

    return html_result

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
    else:
        # File mặc định để test
        test_img_path = "/media/quyet/01DAD374BE175C40/Technical_Draw_Detection/OCR_Table/test_image/image1.png" 

    if os.path.exists(test_img_path):
        extract_table_from_image(test_img_path, output_dir="OCR_Table/output_test_table")
    else:
        print(f"Không tìm thấy file ảnh: {test_img_path}")
        print("Cách dùng: python tabocr_inference.py <đường_dẫn_ảnh_Crop_Table>")