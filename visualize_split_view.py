import os
import cv2
import glob
import yaml
import numpy as np
from ultralytics import YOLO

# ====================================================
# CONFIGURATION
# ====================================================
MODEL_PATH = r'D:\Robo PSM\runs\detect\smart_checkout_model\weights\best.pt'
DATA_YAML  = r'D:\Robo PSM\my_dataset\data.yaml'
OUTPUT_DIR = 'runs/detect/split_view_final_v4'
IOU_THRESHOLD = 0.5

# Box Colors (B, G, R)
COLOR_GT = (255, 255, 255)  # White (Ground Truth)
COLOR_TP = (0, 255, 0)      # Green (True Positive)
COLOR_FP = (0, 0, 255)      # Red (False Positive)
COLOR_FN = (0, 140, 255)    # Orange (False Negative)

# Title Bar Settings
COLOR_TITLE_TEXT = (0, 0, 0)       # Black Text for Titles
COLOR_TITLE_BG   = (200, 200, 200) # Light Gray Bar

def xywh2xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_ground_truth(image_path, classes_names):
    label_path = image_path.replace('images', 'labels').replace(os.path.splitext(image_path)[-1], '.txt')
    if not os.path.exists(label_path):
        label_path = label_path.replace('\\images\\', '\\labels\\').replace('/images/', '/labels/')
    
    img = cv2.imread(image_path)
    if img is None: return [], None
    
    h, w = img.shape[:2]
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    box = xywh2xyxy(cx, cy, bw, bh, w, h)
                    box.append(cls_id)
                    boxes.append(box)
    return boxes, img

def draw_enhanced_box(img, box, color, label=""):
    """Draws box with smart text color adjustment"""
    x1, y1, x2, y2 = box[:4]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position logic
        if y1 - text_h - 5 < 0: 
            text_y = y1 + text_h + 5
            bg_y1, bg_y2 = y1, y1 + text_h + 10
        else:
            text_y = y1 - 5
            bg_y1, bg_y2 = y1 - text_h - 10, y1
            
        # Draw background
        cv2.rectangle(img, (x1, bg_y1), (x1 + text_w + 10, bg_y2), color, -1)
        
        # --- TEXT COLOR LOGIC ---
        # If box is White (GT) OR Green (TP), use BLACK text.
        # Otherwise (Red/Orange), use WHITE text.
        if color == COLOR_GT or color == COLOR_TP:
            txt_col = (0, 0, 0) # Black
        else:
            txt_col = (255, 255, 255) # White
            
        cv2.putText(img, label, (x1 + 5, text_y), font, font_scale, txt_col, thickness)

def draw_panel_title(img, text):
    """Draws title bar with black text"""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), COLOR_TITLE_BG, -1) 
    cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TITLE_TEXT, 2)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    with open(DATA_YAML, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    val_path = data_cfg['val']
    class_names = data_cfg['names']

    if not os.path.isabs(val_path):
        val_path = os.path.join(os.path.dirname(DATA_YAML), val_path)
    
    images = glob.glob(os.path.join(val_path, '*.jpg')) + \
             glob.glob(os.path.join(val_path, 'images', '*.jpg'))
    
    print(f"Processing {len(images)} images... Output: {OUTPUT_DIR}")

    for img_path in images:
        filename = os.path.basename(img_path)
        gt_boxes, original_img = get_ground_truth(img_path, class_names)
        if original_img is None: continue

        results = model.predict(img_path, verbose=False, conf=0.25)
        pred_boxes = []
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            coords.append(cls_id)
            coords.append(conf)
            pred_boxes.append(coords)

        tp_list = []
        fp_list = []
        fn_list = gt_boxes.copy()

        for pred in pred_boxes:
            matched = False
            for i, gt in enumerate(fn_list):
                if compute_iou(pred[:4], gt[:4]) > IOU_THRESHOLD and pred[4] == gt[4]:
                    tp_list.append(pred)
                    fn_list.pop(i)
                    matched = True
                    break
            if not matched:
                fp_list.append(pred)

        # Draw Panels
        img_gt = original_img.copy()
        for box in gt_boxes: draw_enhanced_box(img_gt, box, COLOR_GT, class_names[box[4]])
        draw_panel_title(img_gt, "Ground Truth")

        img_tp = original_img.copy()
        for box in tp_list: draw_enhanced_box(img_tp, box, COLOR_TP, f"{class_names[box[4]]} {box[5]:.2f}")
        draw_panel_title(img_tp, "True Positives")

        img_fp = original_img.copy()
        for box in fp_list: draw_enhanced_box(img_fp, box, COLOR_FP, f"{class_names[box[4]]} {box[5]:.2f}")
        draw_panel_title(img_fp, "False Positives")

        img_fn = original_img.copy()
        for box in fn_list: draw_enhanced_box(img_fn, box, COLOR_FN, f"{class_names[box[4]]} (Missed)")
        draw_panel_title(img_fn, "False Negatives")

        # Stitch & Save
        top_row = np.hstack((img_gt, img_tp))
        bot_row = np.hstack((img_fp, img_fn))
        final_grid = np.vstack((top_row, bot_row))

        save_path = os.path.join(OUTPUT_DIR, f"split_{filename}")
        cv2.imwrite(save_path, final_grid)

    print(f"\n✅ Done! Green boxes now have BLACK text. Check: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()