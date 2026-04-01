import cv2
import os
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Danh sách danh mục class. 
# CHÚ Ý: Cần match với thứ tự các class trong annotation file COCO mà bạn có.
classes = ["PartDrawing", "Note", "Table"]

def get_inference_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Load model Weights đã train
    cfg.MODEL.WEIGHTS = os.path.join("output_model", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    
    # Đặt Threshold để lọc confidence (0.5 là gợi ý, có thể tuỳ chỉnh)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    
    # Yêu cầu CHẠY BẰNG CPU theo mong muốn của user
    cfg.MODEL.DEVICE = "cpu"
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def run_inference(image_path):
    print(f"Đang inference file {image_path} ở chế độ CPU...")
    predictor, cfg = get_inference_model()
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("Lỗi không thể đọc ảnh!")
        return
        
    outputs = predictor(img)
    print("Dự đoán Bounding Box:", outputs["instances"].pred_boxes)
    print("Scores dự đoán:", outputs["instances"].scores)
    print("Class IDs:", outputs["instances"].pred_classes)

    # Hiển thị kết quả bằng Visualizer
    # Tạo một Custom Metadata thay vì đè lên mặc định (tránh lỗi cache của COCO train)
    MetadataCatalog.get("tech_draw_inference").set(thing_classes=classes)
    
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("tech_draw_inference"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    result_img = out.get_image()[:, :, ::-1]
    
    # Lưu file kết quả
    save_path = "result_" + os.path.basename(image_path)
    cv2.imwrite(save_path, result_img)
    print(f"Lưu kết quả tại: {save_path}")

    # Visualize lên màn hình bằng matplotlib (hiệu quả trên Linux/jupyter)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Obeject Detection Result")
    plt.show()

import argparse

if __name__ == "__main__":
    import sys
    # Cho phép truyền đường dẫn ảnh trực tiếp qua command line
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Nếu không truyền, dùng ảnh test mặc định (bạn cần thay đường dẫn bên dưới bằng ảnh có thật)
        test_image = r"/media/quyet/01DAD374BE175C40/Technical_Draw_Detection/Datasets/Dataset_main/valid/25_jpg.rf.e364c072dc880644c0eee4ece910ac8a.jpg" # Edit here

    if os.path.exists(test_image):
        run_inference(test_image)
    else:
        print(f"Không tìm thấy ảnh tại đường dẫn: {test_image}")
        print("Sử dụng lệnh: python Detection/inference.py <đường_dẫn_tới_ảnh>")
