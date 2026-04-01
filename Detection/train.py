import os
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

# Cấu hình log
setup_logger()

# Thiết lập đường dẫn tương đối từ script này ra thư mục gốc để linh hoạt khi chạy lệnh
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

TRAIN_JSON = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage1", "train", "_annotations.coco.json")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage1", "train")
VALID_JSON = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage1", "valid", "_annotations.coco.json")
VALID_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Data_stage1", "valid")

register_coco_instances("tech_draw_train", {}, TRAIN_JSON, TRAIN_DIR)
register_coco_instances("tech_draw_valid", {}, VALID_JSON, VALID_DIR)

def setup_cfg():
    cfg = get_cfg()
    
    # Sử dụng pretrain Faster R-CNN từ ModelZoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Cấu hình dataset
    cfg.DATASETS.TRAIN = ("tech_draw_train",)
    cfg.DATASETS.TEST = ("tech_draw_valid",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Load weights pretrained
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Cấu hình Hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2  # Phụ thuộc vào VRAM GPU, có thể tăng/giảm  
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = 3000    # Số vòng lặp training
    cfg.SOLVER.STEPS = []         # Đặt lại learning rate drop steps
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    # SỐ LƯỢNG CLASSES: PartDrawing, Note, Table (3 class)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    
    # Tắt việc hiển thị mask vì Faster R-CNN chỉ có Bounding Box, trừ khi bạn dùng Mask R-CNN
    cfg.MODEL.MASK_ON = False
    
    # Lưu Model Checkpoints vào thư mục Output của dự án
    cfg.OUTPUT_DIR = "./output_model"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Ensure Device is GPU since you requested GPU for training
    cfg.MODEL.DEVICE = "cuda"
    
    return cfg

if __name__ == "__main__":
    cfg = setup_cfg()
    
    # Bắt đầu training 
    print("Bắt đầu training với Detectron2 trên GPU...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
