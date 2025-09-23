import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import mediapipe as mp
import datetime


# 自動產生新的輸出資料夾 
base_output_folder = r"C:\Users\ytyan\Deeplearning"
folder_name = f"mpii_image_classification_mediapipe_v1"
new_output_folder = os.path.join(base_output_folder, folder_name)

os.makedirs(new_output_folder, exist_ok=True)
print(f" 輸出結果會儲存在: {new_output_folder}")


# 設定來源資料夾 (你分類好的已切割版本)
classified_folder = r"C:\Users\ytyan\Deeplearning\mpii_image_classification_split"

# Mediapipe 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils


# 安全讀圖函數
def safe_read_image(img_path):
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"無法讀取圖片: {img_path}, 錯誤: {e}")
        return None

# 開始疊合 Mediapipe keypoints
for split in ['train', 'val', 'test']:
    split_path = os.path.join(classified_folder, split)
    for class_dir in os.listdir(split_path):
        class_path = os.path.join(split_path, class_dir)
        for img_file in tqdm(os.listdir(class_path), desc=f"{split}/{class_dir}"):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_path, img_file)
            img = safe_read_image(img_path)
            if img is None:
                continue

            # Mediapipe 進行姿勢估計
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            # 畫關鍵點
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            # 儲存新檔案 (自動幫你建立資料夾)
            save_path = os.path.join(new_output_folder, split, class_dir)
            os.makedirs(save_path, exist_ok=True)

            print(f"正在儲存: {os.path.join(save_path, img_file)}")

            cv2.imwrite(os.path.join(save_path, img_file), img)

print("Mediapipe 疊圖完全完成！")
