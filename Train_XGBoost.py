import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_csv('dfi_training_data.csv')
except FileNotFoundError:
    print("找不到 dfi_training_data.csv")
    exit()

X = df.drop('Risk_Level', axis=1)
y = df['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='mlogloss', 
    random_state=42, 
    learning_rate=0.1, 
    max_depth=6
)

print("正在訓練 XGBoost 模型")
xgb_model.fit(X_train, y_train)


print("\n正在進行 5-Fold 交叉驗證")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')

print(f"模型平均準確率 (Mean): {cv_scores.mean() * 100:.2f}%")
print(f"模型穩定度誤差 (STD) : ± {cv_scores.std() * 100:.2f}%")

print("\nXGBoost 分類報告 (Classification Report):")
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))


print("\n正在繪製 SHAP 特徵影響力圖表")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12, 8))

if isinstance(shap_values, list):
    target_class_idx = 2 if len(shap_values) > 2 else 1
    shap.summary_plot(shap_values[target_class_idx], X_test, show=False)
else:
    shap.summary_plot(shap_values, X_test, show=False)

plt.title("SHAP Summary Plot (Risk Level Analysis with Soft Probabilities)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_plot_final.png', dpi=300, bbox_inches='tight')
print("SHAP 圖表已儲存為 shap_summary_plot_final.png")

# ==========================================
# 5. 模型存檔
# ==========================================
joblib.dump(xgb_model, 'xgb_model_final.pkl')
print("\n已儲存為'xgb_model_final.pkl'")
print("="*60)