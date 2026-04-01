import json
import os

def process_annotations_main(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Define new categories
    new_categories = [
        {'id': 0, 'name': 'PartDrawing', 'supercategory': 'none'},
        {'id': 1, 'name': 'Note', 'supercategory': 'none'},
        {'id': 2, 'name': 'Table', 'supercategory': 'none'}
    ]

    # Create mapping from old category id to new category id
    id_mapping = {}
    for cat in data['categories']:
        old_name = cat['name']
        old_id = cat['id']
        
        if old_name == 'PartDrawing':
            id_mapping[old_id] = 0
        elif old_name == 'Note':
            id_mapping[old_id] = 1
        elif old_name == 'Table':
            id_mapping[old_id] = 2
            
    # Update annotations
    new_annotations = []
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in id_mapping:
            ann['category_id'] = id_mapping[old_cat_id]
            new_annotations.append(ann)
            
    data['annotations'] = new_annotations
    data['categories'] = new_categories
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
    print(f"Finished processing {file_path}")

if __name__ == "__main__":
    base_dir = r"d:\Technical_Draw_Detection\Datasets\Dataset_main"
    # Dataset_main has train and valid
    files = [
        os.path.join(base_dir, "train", "_annotations.coco.json"),
        os.path.join(base_dir, "valid", "_annotations.coco.json")
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            process_annotations_main(file_path)
        else:
            print(f"File not found: {file_path}")
