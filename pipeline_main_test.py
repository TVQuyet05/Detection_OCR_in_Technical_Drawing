import os
import cv2
import json
import numpy as np
import logging
from pathlib import Path

# Ngăn chặn hoàn toàn việc gọi các thư viện tối ưu hóa vi xử lý (tránh gây lỗi trên server CPU yếu)
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Cố gắng ép paddle không dùng mkldnn
import paddle
paddle.device.set_device('cpu')
# Ép tắt OneDNN/MKL-DNN trực tiếp qua hàm nội bộ Paddle (đối với 1 số version)
paddle.set_device('cpu') 
try:
    paddle.fluid.core.set_mkldnn_threads(1)
except Exception:
    pass

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
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        # Cấu hình cứng chạy qua CPU theo hệ thống deploy dự kiến
        self.cfg.MODEL.DEVICE = "cpu"
        self.det_predictor = DefaultPredictor(self.cfg)
        
        print("[2/3] Đang nạp mô hình OCR (PP-OCRv4 CPU) cho Note...")
        # LƯU Ý MỚI: Tắt cache shape (ir_optim=False, min_subgraph_size=... )
        self.text_ocr = PaddleOCR(
            use_angle_cls=True, lang='en', use_gpu=False, ocr_version='PP-OCRv4', 
            show_log=False, use_mkldnn=False, enable_mkldnn=False, cpu_threads=1,
            ir_optim=False, use_tensorrt=False
        )
        
        print("[3/3] Đang nạp mô hình cấu trúc Bảng (SLANet CPU) cho Table...")
        # Sử dụng layout=False vì detectron2 đã khoanh vùng "Table" rất chính xác.
        self.table_ocr = PPStructure(
            layout=False, table=True, ocr=True, show_log=False, recovery=True, lang='en', 
            use_gpu=False, use_mkldnn=False, enable_mkldnn=False, cpu_threads=1,
            ir_optim=False, use_tensorrt=False
        )
        
    def process_image(self, image_path):
        """
        Lấy thông tin bounding box, crop ảnh lưu trên RAM, phân luồng OCR, xuất JSON.
        """
        print(f"\n--- Đang xử lý: {image_path} ---")
        img_name = os.path.basename(image_path)
        img_basename = os.path.splitext(img_name)[0]
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không đọc được ảnh từ {image_path}.")
            return None
            
        # Tiền xử lý ảnh: Chuyển sang ảnh xám (Grayscale)
        # PaddleOCR thường chạy tốt trên ảnh xám và Detectron2 COCO Model mặc định chấp nhận 3 Kênh màu, nên ta chuyển grayscale và convert lại 3 kênh.
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
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
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                print(f" -> Bỏ qua đối tượng {i+1} do kích thước quá nhỏ.")
                continue # Box quá nhỏ dễ gây crash
            
            crop_img = np.ascontiguousarray(img[y1:y2, x1:x2])
            
            # Lưu ảnh crop
            crop_dir = os.path.join(self.output_dir, f"crops_{img_basename}")
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f"crop_{i+1}_{class_name}.jpg")
            cv2.imwrite(crop_path, crop_img)
            
            ocr_content = ""
            ocr_confidence = None
            
            # 3. Pipeline: Phân luồng OCR tùy Class
            if class_name == "Note":
                # => Trích xuất dòng chữ
                try:
                    # Tạo bản sao sâu độc lập (tránh lỗi cache RAM khi xử lý Paddle liên tiếp) và ép mảng 3 chiều rõ ràng (Kênh BGR)
                    ocr_crop = np.copy(crop_img) 
                    ocr_result = self.text_ocr.ocr(ocr_crop, cls=True)
                    lines = []
                    conf_list = []
                    if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
                        for box_info in ocr_result[0]:
                            text_val = box_info[1][0]
                            conf_val = float(box_info[1][1])
                            if conf_val > 0.6: # Filter nhẹ
                                lines.append(text_val)
                                conf_list.append(conf_val)
                    ocr_content = "\n".join(lines)
                    if conf_list:
                        ocr_confidence = round(sum(conf_list) / len(conf_list), 3)
                except Exception as e:
                    print(f" -> [CẢNH BÁO] Lỗi khi OCR Note đối tượng {i+1} ({e}). Đang khởi tạo lại Backend...")
                    try:
                        del self.text_ocr
                        import gc
                        gc.collect()
                        self.text_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, ocr_version='PP-OCRv4', show_log=False, use_mkldnn=False, enable_mkldnn=False, cpu_threads=1, ir_optim=False, use_tensorrt=False)
                        
                        ocr_crop = np.copy(crop_img)
                        ocr_result = self.text_ocr.ocr(ocr_crop, cls=True)
                        lines = []
                        conf_list = []
                        if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
                            for box_info in ocr_result[0]:
                                text_val = box_info[1][0]
                                conf_val = float(box_info[1][1])
                                if conf_val > 0.6:
                                    lines.append(text_val)
                                    conf_list.append(conf_val)
                        ocr_content = "\n".join(lines)
                        if conf_list:
                            ocr_confidence = round(sum(conf_list) / len(conf_list), 3)
                        print(f" -> Đã tự phục hồi và OCR Note thành công đối tượng {i+1}!")
                    except Exception as reset_e:
                        print(f" -> [THẤT BẠI] Đã thử lại nhưng vẫn lỗi Note đối tượng {i+1}: {reset_e}")
                        ocr_content = "[OCR Error]"
                        ocr_confidence = None
                
            elif class_name == "Table":
                # => Nhận dạng Bảng & Xuất HTML
                try:
                    tab_crop = np.copy(crop_img) 
                    tab_results = self.table_ocr(tab_crop)
                    html_strings = []
                    for region in tab_results:
                        if region.get('type') == 'table' and 'res' in region:
                            html_strings.append(region['res'].get('html', ''))
                    ocr_content = "\n".join(html_strings)
                    ocr_confidence = None # Tạm để Null cho Table
                except Exception as e:
                    print(f" -> [CẢNH BÁO] Lỗi cấu trúc Table đối tượng {i+1} ({e}). Đang khởi tạo lại Backend...")
                    try:
                        del self.table_ocr
                        import gc
                        gc.collect()
                        self.table_ocr = PPStructure(layout=False, table=True, ocr=True, show_log=False, recovery=True, lang='en', use_gpu=False, use_mkldnn=False, enable_mkldnn=False, cpu_threads=1, ir_optim=False, use_tensorrt=False)
                        
                        tab_crop = np.copy(crop_img)
                        tab_results = self.table_ocr(tab_crop)
                        html_strings = []
                        for region in tab_results:
                            if region.get('type') == 'table' and 'res' in region:
                                html_strings.append(region['res'].get('html', ''))
                        ocr_content = "\n".join(html_strings)
                        ocr_confidence = None
                        print(f" -> Đã tự phục hồi và xử lý Table thành công đối tượng {i+1}!")
                    except Exception as reset_e:
                        print(f" -> [THẤT BẠI] Đã thử lại nhưng vẫn lỗi Table đối tượng {i+1}: {reset_e}")
                        ocr_content = "[Table Extractor Error]"
                        ocr_confidence = None
                
            elif class_name == "PartDrawing":
                # Không ocr part drawing
                ocr_content = "N/A"
            
            # Đóng gói một object
            obj_info = {
                "id": i + 1,
                "class": class_name,
                "confidence": round(score, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": int(y2)},
                "ocr_content": ocr_content,
                "ocr_confidence": ocr_confidence
            }
            
            result_json["objects"].append(obj_info)
            print(f" -> Đã trích xuất xong đối tượng {i+1} [{class_name}]")
            
        json_path = os.path.join(self.output_dir, f"{img_basename}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)
            
        print(f"\n[HOÀN TẤT] Bản vẽ đã được bóc tách và phân tích xong. Kết quả (JSON & hình ảnh Crop) lưu tại: {self.output_dir}")
        
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
