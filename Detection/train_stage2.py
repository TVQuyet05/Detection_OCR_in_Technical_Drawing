import os
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# Cấu hình log
setup_logger()

# Thiết lập đường dẫn tương đối từ script này ra thư mục gốc để linh hoạt khi chạy lệnh
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Trỏ tới dataset Data_stage2 mới tạo
TRAIN_JSON = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage2", "train", "_annotations.coco.json")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage2", "train")
VALID_JSON = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage2", "valid", "_annotations.coco.json")
VALID_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage2", "valid")

register_coco_instances("tech_draw_train_stage2", {}, TRAIN_JSON, TRAIN_DIR)
register_coco_instances("tech_draw_valid_stage2", {}, VALID_JSON, VALID_DIR)

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def setup_cfg():
    cfg = get_cfg()
    
    # Kế thừa cấu hình Faster R-CNN
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Đổi dataset sang Data_stage2
    cfg.DATASETS.TRAIN = ("tech_draw_train_stage2",)
    cfg.DATASETS.TEST = ("tech_draw_valid_stage2",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # LOAD WEIGHT TỪ GIAI ĐOẠN 1 THAY VÌ MODEL ZOO
    cfg.MODEL.WEIGHTS = os.path.join(CURRENT_DIR, "output_model", "model_final.pth")
    
    # Cấu hình Hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00010  # Giảm learning rate xuống một chút vì train tiếp model đã hội tụ một phần
    cfg.SOLVER.MAX_ITER = 540    # Số vòng lặp training stage 2
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    # SỐ LƯỢNG CLASSES: PartDrawing, Note, Table (3 class)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    
    cfg.MODEL.MASK_ON = False
    
    # Lưu Model Checkpoints sang thư mục mới tránh ghi đè model stage 1
    cfg.OUTPUT_DIR = "./output_model_stage2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.MODEL.DEVICE = "cuda"
    
    # Cấu hình đánh giá trên tập validation
    cfg.TEST.EVAL_PERIOD = 60
    
    return cfg

if __name__ == "__main__":
    cfg = setup_cfg()
    
    # Khởi tạo thư mục output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print(f"Bắt đầu tiếp tục training trên Stage 2 với {cfg.MODEL.WEIGHTS}...")
    trainer = CustomTrainer(cfg)
    # resume=False vì đây là quá trình fine-tune với step mới/dataset mới, không phải chạy lại checkpoint bị đứt quãng
    trainer.resume_or_load(resume=False)
    trainer.train()
