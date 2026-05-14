import torch
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # 新增 gridspec 模組
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = [
    "Asparagus", 
    "Averrhoa carambola", 
    "Banana", 
    "Beer", 
    "Broccoli", 
    "Cheese", 
    "Cured Meat", 
    "Ginger", 
    "Grapefruit slices", 
    "Mango", 
    "Orange",
    "Papaya", 
    "Pomelo", 
    "Rice", 
    "Sausage", 
    "Spinach vegetable", 
    "Tomato",
]

food_features = {
    "Grapefruit slices": {"Furanocoumarins": 1, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Pomelo":            {"Furanocoumarins": 1, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Cheese":            {"Furanocoumarins": 0, "Tyramine": 1, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Cured Meat":        {"Furanocoumarins": 0, "Tyramine": 1, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Sausage":           {"Furanocoumarins": 0, "Tyramine": 1, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Beer":              {"Furanocoumarins": 0, "Tyramine": 1, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Broccoli":          {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 1, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Spinach vegetable": {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 1, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Asparagus":         {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 1, "Potassium": 0, "Bleeding_Enhancer": 0},
    "Banana":            {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 1, "Bleeding_Enhancer": 0},
    "Tomato":            {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 1, "Bleeding_Enhancer": 0},
    "Orange":            {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 1, "Bleeding_Enhancer": 0},
    "Averrhoa carambola":{"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 1, "Bleeding_Enhancer": 0}, 
    "Mango":             {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 1},
    "Papaya":            {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 1},
    "Ginger":            {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 1},
    "Rice":              {"Furanocoumarins": 0, "Tyramine": 0, "Vitamin_K": 0, "Potassium": 0, "Bleeding_Enhancer": 0}
}

# 載入模型 (ResNet50 + XGBoost)
print("Loading Model...")
# Resnet50
resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 17)
resnet_model.load_state_dict(torch.load('resnet50_food.pth', map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# XGBoost
xgb_model = joblib.load('xgb_model_final.pkl')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("System open\n")

def analyze_dfi_risk(image_path, patient_drugs, explain_mode=True):
    print("="*50)
    print("Start DFI analysis")
    print("="*50)
    
    # ResNet50(Top-5)
    img_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
    

    top5_probs = top5_probs.cpu().numpy()[0]
    top5_indices = top5_indices.cpu().numpy()[0]
    
    print("Top-5:")
    for i in range(5):
        food_name = class_names[top5_indices[i]]
        print(f"   - {food_name}: {top5_probs[i]*100:.1f}%")


    extracted_features = {"Furanocoumarins": 0.0, "Tyramine": 0.0, "Vitamin_K": 0.0, "Potassium": 0.0, "Bleeding_Enhancer": 0.0}
    
    for prob, idx in zip(top5_probs, top5_indices):
        food_name = class_names[idx]
        for comp in extracted_features.keys():
            extracted_features[comp] += prob * food_features[food_name][comp]


    features_dict = extracted_features.copy()
    features_dict['Drug_Class_Statins']  = 1 if 'Statins' in patient_drugs else 0
    features_dict['Drug_Class_MAOIs']    = 1 if 'MAOIs' in patient_drugs else 0
    features_dict['Drug_Class_Warfarin'] = 1 if 'Warfarin' in patient_drugs else 0
    features_dict['Drug_Class_ACEI']     = 1 if 'ACEI' in patient_drugs else 0
    
    if len(patient_drugs) == 0:
        features_dict['Drug_Class_None'] = 1
    else:
        features_dict['Drug_Class_None'] = 0
        

    df_input = pd.DataFrame([features_dict])
    
    # XGBoost預測
    risk_pred = xgb_model.predict(df_input)[0]
    risk_probs = xgb_model.predict_proba(df_input)[0]
    
    risk_labels = {0: "低風險(安全)", 1: "中風險(注意)", 2: "高風險(危險)"}
    
    print("\n [決策分析]")
    print(f"   輸入化學特徵: {list(np.round(df_input.iloc[0, :5].values, 3))}")
    print(f"   輸入病患用藥: {patient_drugs}")
    print(f"   最終判定: {risk_labels[risk_pred]} (信心水準: {risk_probs[risk_pred]*100:.1f}%)")
    print("="*50)


    if explain_mode:
        

        fig = plt.figure(figsize=(18, 7.5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[4, 1.5])
        
        ax_cam = fig.add_subplot(gs[0, 0])  
        ax_txt = fig.add_subplot(gs[1, 0])  
        ax_shap = fig.add_subplot(gs[:, 1]) 
        

        target_layers = [resnet_model.layer4[-1]]
        cam = GradCAM(model=resnet_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        img_cv = np.array(img_pil)
        img_cv = cv2.resize(img_cv, (224, 224))
        rgb_img = np.float32(img_cv) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        ax_cam.imshow(cam_image)
        ax_cam.set_title("Grad-CAM", fontsize=18, fontweight='bold')
        ax_cam.axis('off')
        

        ax_txt.axis('off')
        top5_str = "Image Recognition Top-5 Probabilities\n"
        for i in range(5):
            food_name = class_names[top5_indices[i]]

            top5_str += f"{i+1}. {food_name.ljust(20)} : {top5_probs[i]*100:>5.1f}%\n"
            

        ax_txt.text(0.5, 0.5, top5_str.strip(), ha='center', va='center', fontsize=17,
                    family='monospace', 
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"))

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(df_input)
        
        plt.sca(ax_shap) 
        

        if len(shap_values.shape) == 3:
           
            shap.plots.waterfall(shap_values[0, :, 2], show=False) 
        else:
            shap.plots.waterfall(shap_values[0], show=False)


        ax_shap.tick_params(axis='y', labelsize=16) 
        ax_shap.tick_params(axis='x', labelsize=16) 
        

        ax_shap.set_xlabel(ax_shap.get_xlabel(), fontsize=17)

  
        for text in ax_shap.texts:
            text.set_fontsize(18) 

        ax_shap.set_title("SHAP Value", fontsize=20, fontweight='bold')

        if risk_pred == 2:
            status_text = "Risk Level: High"
            status_color = "#A81010"  
        elif risk_pred == 1:
            status_text = "Risk Level: Medium"
            status_color = "orange"
        else:
            status_text = "Risk Level: Low"
            status_color = "green"

        
        fig.suptitle(status_text, fontsize=28, fontweight='bold', color=status_color, y=0.98)
        
        plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.1, wspace=0.4, hspace=0.1) 
        
        plt.savefig('xai_demo_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("save 'xai_demo_report.png'")

        

if __name__ == "__main__":
    
    TEST_IMAGE_PATH = './dataset_test/test' 
    
    # 設定有使用藥物
    PATIENT_A_DRUGS = ["MAOIs"]
    
    try:
        analyze_dfi_risk(image_path=TEST_IMAGE_PATH, patient_drugs=PATIENT_A_DRUGS, explain_mode=True)
    except FileNotFoundError:
        print(f"找不到圖片 {TEST_IMAGE_PATH}")