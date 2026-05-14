# 基於雙重可解釋性人工智慧之藥食交互作用預測系統：結合影像辨識與非線性決策模型

## 專案簡介
本專案為國立彰化師範大學資訊管理學系碩士論文之完整實驗程式碼，本研究提出一個結合影像辨識與風險決策之雙重架構系統，目的在預防病患因日常飲食與特定藥物產生交互作用。

## 系統架構
本系統主要分為三大模組：
1. 前端影像特徵擷取 (Image Recognition)：基於 ResNet50 架構，將複雜的日常飲食影像轉化為具體的食材類別機率分佈。
2. 後端風險決策模型 (Risk Decision Model)：透過 XGBoost 綜合前端輸出的特徵與病患用藥紀錄，進行最終的高/低風險判定。
3. 解釋性分析 (Explainable AI)：導入 Grad-CAM 與 SHAP (SHapley Additive exPlanations) 分析，Grad-CAM能夠直觀了解前端影像辨識模型關注區域，SHAP能具體量化各項化學特徵與藥物疊加對風險預警之貢獻度。

## 檔案結構說明
本專案 (`My_project`) 的完整目錄結構與功能說明如下：
My_project/
│
├── dataset/                  # 原始影像資料集 (未進行資料增強)
│   ├── train/                # 訓練集 (內含 17 類食材子資料夾)
│   └── val/                  # 驗證集 (內含 17 類食材子資料夾)
│
├── dataset_augmented/        # 擴充影像資料集 (經資料增強處理，內含 17 類食材子資料夾)
│
├── dataset_test/             # 獨立測試集 (存放模型最終測試用之影像)
│
├── demo/                     # 系統展示與操作介面相關程式
│
├── crawler.py                # 網路爬蟲程式 (用於自動化蒐集食材影像資料)
├── train_resnet50.py         # 前端影像辨識模型 (ResNet50) 訓練程式
├── train_xgboost.py          # 後端風險決策模型 (XGBoost) 訓練程式
└── test_interference.py      # 後端決策模型之抗干擾性與穩定性測試

## 執行環境與核心套件
* Python 3.11.9
* PyTorch 
* XGBoost / Scikit-learn / LightGBM 
* SHAP / Grad-CAM 
* OpenCV / PIL 
* Pandas / NumPy
