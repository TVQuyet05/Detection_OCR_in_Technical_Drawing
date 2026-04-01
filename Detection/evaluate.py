import os
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Cấu hình log
setup_logger()

# Thiết lập đường dẫn thư mục gốc tương đối
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Đường dẫn đến tập validation
VALID_JSON = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage1", "valid", "_annotations.coco.json")
VALID_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage1", "valid")

# Nếu tập dataset chưa được đăng ký trong session hiện tại, thực hiện đăng ký
try:
    register_coco_instances("tech_draw_valid", {}, VALID_JSON, VALID_DIR)
except Exception as e:
    print(f"Dataset đã được đăng ký hoặc gặp lỗi: {e}")

def get_eval_cfg():
    cfg = get_cfg()
    # Đọc base config giống lúc train
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Cấu hình dataset
    cfg.DATASETS.TEST = ("tech_draw_valid",)
    
    # Cấu hình kiến trúc giống quá trình training 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.DEVICE = "cuda"  # hoặc "cpu" nếu chạy test không có GPU
    
    # Load weights model đã train thành công (file model_final.pth nằm trong thư mục output_model)
    cfg.MODEL.WEIGHTS = os.path.join(CURRENT_DIR, "output_model", "model_final.pth")
    
    # Thiết lập ngưỡng confidence score để evaluate (có thể tinh chỉnh, vd: lấy box có tin cậy > 50%)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
    
    return cfg

def main():
    cfg = get_eval_cfg()
    
    print("Đang load model...")
    predictor = DefaultPredictor(cfg)
    
    print("Bắt đầu Evaluate tính mAP...")
    # COCOEvaluator sử dụng format của MS COCO để đo lường mAP
    evaluator = COCOEvaluator("tech_draw_valid", output_dir=os.path.join(CURRENT_DIR, "output_model"))
    val_loader = build_detection_test_loader(cfg, "tech_draw_valid")
    
    # Thực thi tính metrics
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    main()
