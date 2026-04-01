import os
import cv2
import json
import numpy as np
import logging
from pathlib import Path

# Cần set biến môi trường này trước khi import paddle để tránh lỗi OpenBLAS/OneDNN trên CPU
os.environ['FLAGS_use_mkldnn'] = '0'

# Detectron2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Paddle OCR & SLANet import
from paddleocr import PaddleOCR, PPStructure

# Ẩn bớt log quá dày của PaddleOCR
logging.getLogger("ppocr").setLevel(logging.ERROR)

class TechnicalDrawingPipeline:
    def __init__(self, model_weights="output_model/model_final.pth", output_dir="Pipeline_Output"):
        """
        Khởi tạo Pipeline tích hợp Detection và OCR.
        :param model_weights: Đường dẫn tới weights Detectron2 đã train xong.
        :param output_dir: Thư mục chứa các ảnh crop và file json kết quả.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Thứ tự class phải KHỚP với json COCO annotations mà bạn đã train
        self.classes = ["PartDrawing", "Note", "Table"]
        
        print("\n[1/3] Đang nạp mô hình Object Detection (Detectron2 CPU)...")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_weights
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.classes)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        # Cấu hình cứng chạy qua CPU theo hệ thống deploy dự kiến
        self.cfg.MODEL.DEVICE = "cpu"
        self.det_predictor = DefaultPredictor(self.cfg)
        
        print("[2/3] Đang nạp mô hình OCR (PP-OCRv4 CPU) cho Note...")
        self.text_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, ocr_version='PP-OCRv4', show_log=False)
        
        print("[3/3] Đang nạp mô hình cấu trúc Bảng (SLANet CPU) cho Table...")
        # Layout=False vì ta đã dùng Detectron crop đúng vùng Table rồi.
        self.table_ocr = PPStructure(layout=False, table=True, ocr=True, show_log=False, recovery=True, lang='en', device='cpu')
        
    def process_image(self, image_path):
        """
        Lấy thông tin bounding box, crop ảnh, phân luồng OCR, xuất JSON.
        """
        print(f"\n--- Đang xử lý: {image_path} ---")
        img_name = os.path.basename(image_path)
        img_basename = os.path.splitext(img_name)[0]
        
        # Thư mục chứa từng component crop ra riêng biệt
        crop_dir = os.path.join(self.output_dir, f"crops_{img_basename}")
        os.makedirs(crop_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không đọc được ảnh từ {image_path}.")
            return None
            
        h_img, w_img, _ = img.shape
            
        # 1. Pipeline: Detect vùng
        outputs = self.det_predictor(img)
        instances = outputs["instances"].to("cpu")
        
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        class_ids = instances.pred_classes.numpy()
        
        result_json = {
            "image": img_name,
            "objects": []
        }
        
        # 2. Pipeline: Duyệt qua từng đối tượng và crop
        for i in range(len(boxes)):
            box = boxes[i]
            score = float(scores[i])
            class_id = int(class_ids[i])
            class_name = self.classes[class_id]
            
            x1, y1, x2, y2 = map(int, box)
            # Chuẩn hoá kích thước tránh lỗi tràn index do viền
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue # Box phi lý
            
            crop_img = img[y1:y2, x1:x2]
            
            # Đặt tên lưu và lưu crop
            crop_filename = f"{i+1:03d}_{class_name}_conf{score:.2f}.jpg"
            crop_path = os.path.join(crop_dir, crop_filename)
            cv2.imwrite(crop_path, crop_img)
            
            ocr_content = ""
            
            # 3. Pipeline: Phân luồng OCR tùy Class
            if class_name == "Note":
                # => Trích xuất dòng chữ
                ocr_result = self.text_ocr.ocr(crop_img, cls=True)
                lines = []
                if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
                    for box_info in ocr_result[0]:
                        text_val = box_info[1][0]
                        conf_val = box_info[1][1]
                        if conf_val > 0.6: # Filter nhẹ
                            lines.append(text_val)
                ocr_content = "\n".join(lines)
                
            elif class_name == "Table":
                # => Nhận dạng Bảng & Xuất HTML
                tab_results = self.table_ocr(crop_img)
                # tab_results trả list có region='table' hoặc đôi khi trống
                html_strings = []
                for region in tab_results:
                    if region.get('type') == 'table' and 'res' in region:
                        html_strings.append(region['res'].get('html', ''))
                ocr_content = "\n".join(html_strings)
                
            elif class_name == "PartDrawing":
                # Không ocr part drawing
                ocr_content = "N/A"
            
            # Đóng gói một object
            obj_info = {
                "id": i + 1,
                "class": class_name,
                "confidence": round(score, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": int(y2)},
                "ocr_content": ocr_content
            }
            
            result_json["objects"].append(obj_info)
            print(f" -> Đã trích xuất xong đối tượng {i+1} [{class_name}]")
            
        # 4. Ghi tổng hợp ra File Json
        json_path = os.path.join(self.output_dir, f"{img_basename}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
            
        print(f"\n[HOÀN TẤT] Bản vẽ đã được bóc tách.")
        print(f" - Toàn bộ Ảnh Crop được lưu tại  : {crop_dir}")
        print(f" - JSON cấu trúc tổng được lưu tại: {json_path}")
        
        return result_json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-To-End Pipeline Technical Draw Detection.")
    parser.add_argument("--image", type=str, default="/media/quyet/01DAD374BE175C40/Technical_Draw_Detection/Datasets/Dataset_main/valid/11_jpg.rf.c64c8111f7bf3bbbe86aa002134ab98b.jpg", help="Đường dẫn đến bản vẽ cần xử lý.")
    parser.add_argument("--weights", type=str, default="Detection/output_model/model_final.pth", help="Đường dẫn weights detectron2.")
    parser.add_argument("--outdir", type=str, default="Pipeline_Output", help="Thư mục ghi kết quả json và crop.")
    args = parser.parse_args()
    
    # Kiểm tra weights
    if not os.path.exists(args.weights):
        print(f"Không tìm thấy model weights tại: {args.weights}")
        print("Vui lòng cập nhật đúng tham số --weights.")
        exit(1)
        
    if not os.path.exists(args.image):
        print(f"Không tìm thấy ảnh tại: {args.image}")
        exit(1)

    # Khởi tạo pipeline
    pipeline = TechnicalDrawingPipeline(model_weights=args.weights, output_dir=args.outdir)
    pipeline.process_image(args.image)
