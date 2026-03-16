# TODO



"""
coco2txt
"""

import json
import os
from collections import defaultdict

train_datasets_path     = r"ITSDT/images"
val_datasets_path       = r"ITSDT/images"

train_annotation_path   = r"ITSDT/instances_train2017.json"
val_annotation_path     = r"ITSDT/instances_test2017.json"

train_output_path       = r"ITSDT/coco_train_ITSDT.txt"
val_output_path         = r"ITSDT/coco_val_ITSDT.txt"

def get_path(images, id):
    for image in images:
        if id == image["id"]:
            return image['file_name']
    
def load_json_with_fallback(path):
    for enc in ('utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                text = f.read()
            s = text.lstrip()
            if not s or s[0] not in ('{', '['):
                raise ValueError("not JSON text")
            return json.loads(text)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    b = open(path, 'rb').read()
    first = next((c for c in b if c not in (9,10,13,32)), None)
    if first in (0x7b, 0x5b):  # '{' or '['
        for enc in ('utf-8', 'latin-1'):
            try:
                return json.loads(b.decode(enc))
            except Exception:
                continue
    raise RuntimeError(f"{path} 似乎为二进制文件（首字节 0x{first:02x}），不是 JSON 文本，请检查并重新生成")
        
        
if __name__ == "__main__":
    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = load_json_with_fallback(train_annotation_path)
    data        = f
    
    images = data['images']
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(train_datasets_path, get_path(images, id))
        cat = ant['category_id'] - 1
        name_box_id[name].append([ant['bbox'], cat])

    f = open(train_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    

    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = load_json_with_fallback(val_annotation_path)
    data        = f
    
    images = data['images']
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(train_datasets_path, get_path(images, id))
        cat = ant['category_id']
        cat = cat - 1
        name_box_id[name].append([ant['bbox'], cat])

    f = open(val_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    