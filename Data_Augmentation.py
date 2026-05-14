import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from tqdm import tqdm


INPUT_DIR = './dataset/train'            
OUTPUT_DIR = 'dataset_augmented'
AUG_PER_IMAGE = 8 
TARGET_SIZE = (224, 224)          


thesis_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE),                           
    transforms.RandomRotation(degrees=30),                    
    transforms.RandomHorizontalFlip(p=0.5),                   
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2)      
])


resize_only = transforms.Resize(TARGET_SIZE)


def augment_dataset():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到輸入資料夾：{INPUT_DIR}")
        return

    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    print(f"資料增強，共 {len(classes)} 個類別")

    for cls in classes:
        input_class_path = os.path.join(INPUT_DIR, cls)
        output_class_path = os.path.join(OUTPUT_DIR, cls)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
            
        images = os.listdir(input_class_path)
        
        for img_name in tqdm(images, desc=f"處理 {cls}"):
            img_path = os.path.join(input_class_path, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                
                base_img = resize_only(img)
                base_img.save(os.path.join(output_class_path, f"org_{img_name}"))
                
                
                for i in range(AUG_PER_IMAGE):
                    aug_img = thesis_transforms(img)
                    aug_img.save(os.path.join(output_class_path, f"aug_{i}_{img_name}"))
                    
            except (UnidentifiedImageError, OSError):
                pass 

    print("完成！")

if __name__ == "__main__":
    augment_dataset()