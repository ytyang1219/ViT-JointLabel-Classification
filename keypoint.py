import os
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm
from PIL import Image

# ================================================
# 設定檔案路徑
# ================================================
mat_path = r"C:\Users\ytyan\Deeplearning\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat"
classified_folder = r"C:\Users\ytyan\Deeplearning\mpii_image_classification_split"
new_output_folder = r"C:\Users\ytyan\Deeplearning\mpii_image_classification_keypoints"

# ================================================
# 讀取 .mat 標註檔案
# ================================================
mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
release = mat['RELEASE']
annolist = release.annolist

# ================================================
# 建立 keypoints 對應字典（每張圖可有多個人）
# ================================================
keypoints_dict = {}

for idx in tqdm(range(len(annolist)), desc="Building keypoints dict"):
    ann = annolist[idx]
    img_name = os.path.basename(ann.image.name).lower()
    all_joints = []
    if hasattr(ann, 'annorect') and ann.annorect is not None:
        annorect = ann.annorect
        if not isinstance(annorect, (list, np.ndarray)):
            annorect = [annorect]
        for rect in annorect:
            if hasattr(rect, 'annopoints') and rect.annopoints is not None:
                annopoints = rect.annopoints
                # 兼容各種型態
                if hasattr(annopoints, 'point'):
                    points = annopoints.point
                    if not isinstance(points, (list, np.ndarray)):
                        points = [points]
                elif isinstance(annopoints, (list, np.ndarray)):
                    points = annopoints
                else:
                    points = []
                joints = []
                for p in points:
                    try:
                        x = p.x
                        y = p.y
                        joints.append((x, y))
                    except Exception as e:
                        continue
                if joints:
                    all_joints.append(joints)
    if all_joints:
        keypoints_dict[img_name] = all_joints

print(f"共整理出 {len(keypoints_dict)} 張有標註 keypoints 的圖像")

# ================================================
# 安全讀圖函數
# ================================================
def safe_read_image(img_path):
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"Pillow 也無法讀取圖片: {img_path}, Error: {e}")
        return None

# ================================================
# 疊合 keypoints 到分類後的圖片上
# ================================================
for split in ['train', 'val', 'test']:
    split_path = os.path.join(classified_folder, split)
    if not os.path.exists(split_path):
        print(f" Split 資料夾不存在: {split_path}")
        continue
    for class_dir in os.listdir(split_path):
        class_path = os.path.join(split_path, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_file in tqdm(os.listdir(class_path), desc=f"{split}/{class_dir}"):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_path, img_file)
            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                print(f"檔案不存在或為空檔: {img_path}")
                continue

            img = safe_read_image(img_path)
            if img is None:
                continue

            img_file_lower = img_file.lower()
            # 疊合所有人的 keypoints
            if img_file_lower in keypoints_dict:
                for person_joints in keypoints_dict[img_file_lower]:
                    for (x, y) in person_joints:
                        try:
                            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                        except Exception as e:
                            print(f" 畫點失敗: {img_file}, x={x}, y={y}, Error: {e}")
                            continue

            save_path = os.path.join(new_output_folder, split, class_dir)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, img_file), img)

print("批次疊圖轉換完成，所有圖片已儲存")
