import json
import os
import shutil

def merge_coco_datasets(input_dirs, output_dir, prefix_mapping):
    print(f"Bắt đầu gộp vào {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_images = []
    merged_annotations = []
    
    # Categories chuẩn
    new_categories = [
        {'id': 0, 'name': 'PartDrawing', 'supercategory': 'none'},
        {'id': 1, 'name': 'Note', 'supercategory': 'none'},
        {'id': 2, 'name': 'Table', 'supercategory': 'none'}
    ]
    
    global_image_id = 0
    global_ann_id = 0
    
    for input_dir, prefix in zip(input_dirs, prefix_mapping):
        ann_file = os.path.join(input_dir, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            print(f"Bỏ qua {input_dir}, không tìm thấy file annotation.")
            continue
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        image_id_map = {} # map old_id -> new_id
        
        for img in data['images']:
            old_id = img['id']
            global_image_id += 1
            new_id = global_image_id
            image_id_map[old_id] = new_id
            
            old_filename = img['file_name']
            
            # Đổi tên file để tránh trùng lặp giữa các dataset
            new_filename = f"{prefix}_{old_filename}"
            
            img['id'] = new_id
            img['file_name'] = new_filename
            
            # Phải copy file ảnh sang thư mục mới
            src_img_path = os.path.join(input_dir, old_filename)
            dst_img_path = os.path.join(output_dir, new_filename)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Cảnh báo: Không tìm thấy ảnh {src_img_path}")
                
            merged_images.append(img)
            
        for ann in data['annotations']:
            global_ann_id += 1
            ann['id'] = global_ann_id
            ann['image_id'] = image_id_map[ann['image_id']]
            merged_annotations.append(ann)
            
    merged_data = {
        'info': {'description': 'Merged Dataset Stage1'},
        'licenses': [],
        'categories': new_categories,
        'images': merged_images,
        'annotations': merged_annotations
    }
    
    merged_ann_file = os.path.join(output_dir, '_annotations.coco.json')
    with open(merged_ann_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f)
        
    print(f"=> Hoàn tất gộp {len(input_dirs)} thư mục. Tổng số ảnh: {len(merged_images)}, Tổng số annotations: {len(merged_annotations)}\n")

if __name__ == '__main__':
    base_dir = r"d:\Technical_Draw_Detection\Datasets"
    
    # 1. Gộp Train
    train_inputs = [
        os.path.join(base_dir, "Data_cad", "train"),
        os.path.join(base_dir, "Data_tech", "train")
    ]
    train_prefixes = ["cad_train", "tech_train"]
    train_output = os.path.join(base_dir, "Data_stage1", "train")
    merge_coco_datasets(train_inputs, train_output, train_prefixes)
    
    # 2. Gộp Valid
    valid_inputs = [
        os.path.join(base_dir, "Data_cad", "valid"),
        os.path.join(base_dir, "Data_cad", "test"),
        os.path.join(base_dir, "Data_tech", "valid")
    ]
    valid_prefixes = ["cad_valid", "cad_test", "tech_valid"]
    valid_output = os.path.join(base_dir, "Data_stage1", "valid")
    merge_coco_datasets(valid_inputs, valid_output, valid_prefixes)
