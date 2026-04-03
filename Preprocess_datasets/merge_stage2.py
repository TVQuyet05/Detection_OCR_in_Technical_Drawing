import json
import os
import shutil
import random
import cv2

def create_dataset_split(inputs_info, output_dir):
    """
    inputs_info is a list of tuples: (source_dir, prefix, max_images)
    """
    print(f"Bắt đầu tạo Dataset vào: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_images = []
    merged_annotations = []
    
    # Category chuẩn
    new_categories = [
        {'id': 0, 'name': 'PartDrawing', 'supercategory': 'none'},
        {'id': 1, 'name': 'Note', 'supercategory': 'none'},
        {'id': 2, 'name': 'Table', 'supercategory': 'none'}
    ]
    
    global_image_id = 0
    global_ann_id = 0
    
    for input_dir, prefix, max_images in inputs_info:
        ann_file = os.path.join(input_dir, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            print(f"Bỏ qua {input_dir}, không tìm thấy file annotation.")
            continue
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        images_list = data['images']
        random.seed(42) # Fix seed để kết quả lấy randome luôn giống nhau mỗi khi chạy
        
        if max_images is not None and len(images_list) > max_images:
            images_list = random.sample(images_list, max_images)
            
        selected_image_ids = {img['id'] for img in images_list}
        
        image_id_map = {} # map old_id -> new_id
        
        for img in images_list:
            old_id = img['id']
            global_image_id += 1
            new_id = global_image_id
            image_id_map[old_id] = new_id
            
            old_filename = img['file_name']
            new_filename = f"{prefix}_{old_filename}"
            
            src_img_path = os.path.join(input_dir, old_filename)
            dst_img_path = os.path.join(output_dir, new_filename)
            
            # Tạo bản copy tránh chỉnh sửa file gốc khi load
            img_copy = img.copy()
            img_copy['id'] = new_id
            img_copy['file_name'] = new_filename
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
                # Fix lỗi khác kích thước giữa ảnh thực tế và file json
                img_cv2 = cv2.imread(src_img_path)
                if img_cv2 is not None:
                    h, w = img_cv2.shape[:2]
                    img_copy['height'] = h
                    img_copy['width'] = w
            else:
                print(f"Cảnh báo: Không tìm thấy ảnh {src_img_path}")
                
            merged_images.append(img_copy)
            
        for ann in data['annotations']:
            if ann['image_id'] in selected_image_ids:
                global_ann_id += 1
                ann_copy = ann.copy()
                ann_copy['id'] = global_ann_id
                ann_copy['image_id'] = image_id_map[ann['image_id']]
                
                # Đồng bộ Category ID mapping nếu cần (Hiện tại giả định chung bộ chuẩn 0, 1, 2)
                merged_annotations.append(ann_copy)
            
    merged_data = {
        'info': {'description': 'Merged Dataset Stage 2'},
        'licenses': [],
        'categories': new_categories,
        'images': merged_images,
        'annotations': merged_annotations
    }
    
    merged_ann_file = os.path.join(output_dir, '_annotations.coco.json')
    with open(merged_ann_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False)
        
    print(f"=> Hoàn tất! Đã lưu {len(merged_images)} ảnh và {len(merged_annotations)} annotations vào {output_dir}.\n")

if __name__ == '__main__':
    # Workspace Dir
    base_dir = r"/media/quyet/01DAD374BE175C40/Technical_Draw_Detection/Datasets"
    
    # 1. Tạo folder Train
    train_inputs = [
        (os.path.join(base_dir, "Dataset_main", "train"), "main_tr", 90),
        (os.path.join(base_dir, "Data_stage1", "valid"), "old_val", 90)
    ]
    train_output = os.path.join(base_dir, "Data_stage2", "train")
    create_dataset_split(train_inputs, train_output)
    
    # 2. Tạo folder Valid
    valid_inputs = [
        (os.path.join(base_dir, "Dataset_main", "valid"), "main_val", 27)
    ]
    valid_output = os.path.join(base_dir, "Data_stage2", "valid")
    create_dataset_split(valid_inputs, valid_output)
