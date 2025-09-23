# 多關節點標註結合影像之分類效能比較
**以 MPII 與 Mediapipe 關節點為例**  
組員：楊詠婷、林鼎鈞、周姵妍

---

## 專案介紹
隨著影像分類技術日益成熟，挑戰在於如何更快速且準確地判別多類別影像。本研究基於 **Vision Transformer (ViT)** 架構，針對以下三種輸入方式進行比較：

1. **原始影像 (Image only)**  
2. **MPII 16個關節標註 + 影像**  
3. **Mediapipe 33個關節點 + 影像**  

比較面向包含分類效能、準確度與收斂效果，藉此探討「疊加節點設計」對於多類別影像分類的影響。

---

## 研究方法

### 資料集處理
- 使用 **MPII Human Pose Dataset**，並針對類別數量不均問題進行篩選。  
- 保留影像張數介於 160–175 的 7 類活動（如 ballet、bicycling、rowing 等）。  
- 依 80% / 10% / 10% 比例分為訓練集、驗證集與測試集。  
- 另外建立 Mediapipe 偵測關節點後的資料集。

### 模型架構
- 使用 **Vision Transformer (ViT)** 作為基礎。  
- 將影像切分為 patch → Transformer Self-Attention 特徵抽取 → 最終分類。  
- 訓練參數：Epoch=50、Batch=16、Learning rate=2e-5。

### 模型訓練與分析
- 分別對三種資料集訓練模型，並比較 **Loss Curve、混淆矩陣、精準度 (mAP、F1-score)** 等指標。

---

## 實驗結果

### 分類表現
- **Image only**：mAP=0.95、F1-score=0.975 → 整體最佳。  
- **MPII**：分類錯誤數量約為原始影像的兩倍，精度下降。  
- **Mediapipe**：雖未超越 Image only，但保持次高準確度。  

### 收斂曲線
- 三種方式皆在第 10 個 epoch 左右完成快速收斂。  
- 訓練過程中未出現明顯過擬合，表現穩定。

---

## 遇到的問題與解決方案
- **MPII 資料集未整理** → 撰寫程式自動分類、建立 train/val/test 資料夾。  
- **OpenCV 讀圖失敗** → 改用純英文路徑命名。  
- **.mat 格式不一致** → 使用 `hasattr()`、`isinstance()` 與 `try-except` 處理。  
- **GPU 記憶體不足** → 將 batch size 由 64 降至 16。  
- **預訓練權重不相符** → 採用 Hugging Face `vit-base-patch16-224-in21k` 原始架構，避免 mismatch。  
- **Mediapipe 限制** → 僅支援單人偵測，但提供更多節點與骨架連線。  

---

## 結論
- 原始影像 (Image only) 已具備良好分類效果。  
- 疊加節點資訊反而導致部分影像特徵被覆蓋，影響訓練效能。  
- 節點資訊未必適合作為直接輸入，但可作為後續錯誤修正或輔助資訊。  

---

## 參考資料
- MPII Human Pose Dataset: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download  
- Vision Transformer 原始論文：[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)  
- 相關部落格與筆記（CSDN、Medium）

---

## 分工表
 **楊詠婷**：MPII/Mediapipe 關節點疊圖程式、資料集建立、mAP/F1 分析、報告撰寫  
- **林鼎鈞**：研究主題發想、訓練與驗證、Loss Curve 分析、報告撰寫  
- **周姵妍**：資料前處理與 Dataloader、混淆矩陣分析、結論、報告撰寫  

