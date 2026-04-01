import os
import cv2
import logging
import numpy as np
import gradio as gr
from pipeline_main import TechnicalDrawingPipeline

# Thiết lập log level
logging.basicConfig(level=logging.INFO)

# Đường dẫn mặc định của mô hình lúc deploy (cần trỏ đúng path model của bạn)
MODEL_WEIGHTS = "Detection/output_model/model_final.pth" 

# Cờ để kiểm tra đã load mô hình chưa
pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is None:
        if not os.path.exists(MODEL_WEIGHTS):
            raise FileNotFoundError(f"Không tìm thấy model weights tại: {MODEL_WEIGHTS}. Vui lòng kiểm tra lại trước khi deploy.")
        # Khởi tạo mô hình chỉ 1 lần duy nhất cho toàn bộ app
        pipeline = TechnicalDrawingPipeline(model_weights=MODEL_WEIGHTS, output_dir="Pipeline_Output")
    return pipeline

# Bảng màu hiển thị (BGR array -> RGB array) khi vẽ OpenCV
COLOR_MAP = {
    "PartDrawing": (255, 0, 0),     # Red (RGB)
    "Note": (0, 255, 0),            # Green (RGB)
    "Table": (0, 0, 255)            # Blue (RGB)
}

def draw_boxes(image_path, json_data):
    """
    Load lại ảnh và vẽ các Bounding box minh hoạ.
    """
    img = cv2.imread(image_path)
    # Chuyển BGR (OpenCV) sang RGB (Gradio)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for obj in json_data.get("objects", []):
        class_name = obj["class"]
        conf = obj["confidence"]
        box = obj["bbox"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        
        color = COLOR_MAP.get(class_name, (255, 255, 0)) # Default Yellow
        
        # Vẽ Khung
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # Vẽ Nhãn (Background mờ đằng sau nhãn cho dễ nhìn)
        label = f"{class_name} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    return img

def extract_crops(image_path, json_data):
    """
    Cắt ảnh ra thành các Crops và nối lại thành 1 ảnh duy nhất (Numpy Array)
    để hiển thị an toàn trên giao diện giống hệt cơ chế của draw_boxes.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_imgs = []
    
    for obj in json_data.get("objects", []):
        class_name = obj["class"]
        conf = obj["confidence"]
        box = obj["bbox"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        
        # Đề phòng tọa độ vượt quá kích thước hoặc lỗi
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            c_img = img[y1:y2, x1:x2].copy()
            label = f"#{obj['id']} - {class_name} ({conf:.2f})"
            
            # Thêm 1 thanh header đen để ghi chữ Header
            bar_height = 40
            c_h, c_w, _ = c_img.shape
            bar = np.zeros((bar_height, c_w, 3), dtype=np.uint8)
            cv2.putText(bar, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            combined = np.vstack((bar, c_img))
            crop_imgs.append(combined)
            
    if not crop_imgs:
        # Nếu Không có object nào
        return np.ones((100, 400, 3), dtype=np.uint8) * 255
        
    # Chuẩn hoá kích thước width lớn nhất để dùng np.vstack (nối ảnh)
    max_w = max(c.shape[1] for c in crop_imgs)
    padded_crops = []
    
    for c in crop_imgs:
        c_w = c.shape[1]
        pad_w = max_w - c_w
        if pad_w > 0:
            c = cv2.copyMakeBorder(c, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        padded_crops.append(c)
        
    # Tạo 1 bức ảnh ghép tất cả dọc từ trên xuống dưới
    final_img = padded_crops[0]
    for c in padded_crops[1:]:
        separator = np.ones((20, max_w, 3), dtype=np.uint8) * 150 # Gạch phân cách
        final_img = np.vstack((final_img, separator, c))
        
    return final_img

def format_ocr_html(json_data):
    """
    Format kết quả OCR (text và table layout) sang định dạng HTML đẹp mắt.
    """
    html_content = "<div>"
    
    objects = json_data.get("objects", [])
    if not objects:
        return "<i>Không tìm thấy đối tượng nào.</i>"
        
    for obj in objects:
        cls_name = obj["class"]
        ocr_text = obj.get("ocr_content", "")
        obj_id = obj["id"]
        
        if cls_name == "Note":
            html_content += f"<h3>📝 Note #{obj_id}</h3>"
            if ocr_text.strip() == "":
                html_content += "<p><i>(Trống)</i></p>"
            else:
                html_content += f"<pre style='background-color:#f4f4f4; padding:10px; border-radius:5px;'>{ocr_text}</pre>"
                
        elif cls_name == "Table":
            html_content += f"<h3>📊 Table #{obj_id} (HTML)</h3>"
            if ocr_text.strip() == "":
                 html_content += "<p><i>(Không giải mã được cấu trúc HTML)</i></p>"
            else:
                # Add table border styling inline for better preview
                styled_table = ocr_text.replace("<html><body>", "").replace("</body></html>", "")
                styled_table = styled_table.replace("<table", "<table border='1' style='border-collapse: collapse; min-width: 50%;'")
                html_content += styled_table
                
    html_content += "</div>"
    return html_content

def process_ui(image_path):
    if image_path is None:
        return None, {}, None, "Vui lòng tải lên một ảnh."
    
    try:
        pipe = load_pipeline()
    except Exception as e:
         return None, {"error": str(e)}, None, f"<span style='color:red'>{str(e)}</span>"

    # Gọi inference chính
    results = pipe.process_image(image_path)
    
    if not results:
        return None, {}, None, "Có lỗi khi phân tích ảnh này."
        
    # Tạo kết quả thị giác hóa (Visualization)
    annotated_img = draw_boxes(image_path, results)
    
    # Cắt ảnh crop object & Gộp thành 1 ảnh numpy array duy nhất !
    collage_crops_img = extract_crops(image_path, results)
    
    # Tạo HTML render OCR
    ocr_html = format_ocr_html(results)
    
    return annotated_img, results, collage_crops_img, ocr_html


# Xây dựng giao diện web với Gradio
with gr.Blocks(title="Technical Draw Detection") as demo:
    gr.Markdown("# 📐 Technical Drawing Layout Analysis")
    gr.Markdown("Nhận diện bố cục (PartDrawing, Note, Table) trên bản vẽ kỹ thuật, trích xuất text (PP-OCR) và mô hình hóa Bảng (SLANet). **Optimized for CPU.**")
    
    with gr.Row():
        # Cột Input
        with gr.Column(scale=4):
            input_image = gr.Image(type="filepath", label="Upload ảnh bản vẽ")
            submit_btn = gr.Button("🚀 Gửi / Detect", variant="primary")
            
        # Cột Output (Visual & JSON)
        with gr.Column(scale=6):
            output_image = gr.Image(label="Visualization")
            json_output = gr.JSON(label="JSON Panel (Thông tin toạ độ & Meta)")
            
    # Hàng ở dưới full length cho Crops (Dùng chung tính năng gr.Image)
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown("## 🖼️ Các đối tượng được tách ra (Cropped Objects)")
            crop_collage_ui = gr.Image(label="Cropped Objects Visualization")

    # Hàng ở dưới full length cho OCR
    with gr.Row():
         with gr.Column(scale=10):
            gr.Markdown("## 🖹 OCR & Table HTML Panel")
            ocr_output = gr.HTML(label="OCR results")
            
    # Gán sự kiện
    submit_btn.click(
        fn=process_ui,
        inputs=[input_image],
        outputs=[output_image, json_output, crop_collage_ui, ocr_output]
    )

if __name__ == "__main__":
    # Launch server
    # Nếu deploy HuggingSpaces thì để mặc định 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
