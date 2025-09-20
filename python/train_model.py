"""
مشروع تصنيف الأمراض الجلدية باستخدام PyTorch
Skin Disease Classification using PyTorch

هذا الملف يحتوي على الكود الكامل لتدريب نموذج تصنيف الأمراض الجلدية
باستخدام مجموعة بيانات HAM10000 ونموذج ResNet50 مع Transfer Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# إعداد الجهاز (GPU أو CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'استخدام الجهاز: {device}')

class SkinDiseaseDataset(Dataset):
    """
    فئة مخصصة لتحميل بيانات الأمراض الجلدية
    Custom Dataset class for skin disease images
    """
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        
        # تعريف الفئات السبع للأمراض الجلدية
        self.classes = {
            'nv': 0,    # Melanocytic nevi
            'mel': 1,   # Melanoma
            'bkl': 2,   # Benign keratosis-like lesions
            'bcc': 3,   # Basal cell carcinoma
            'akiec': 4, # Actinic keratoses
            'vasc': 5,  # Vascular lesions
            'df': 6     # Dermatofibroma
        }
        
        # أسماء الفئات باللغة العربية
        self.class_names = {
            0: 'الشامات الصبغية',
            1: 'الورم الميلانيني الخبيث',
            2: 'الآفات الحميدة الشبيهة بالتقرن',
            3: 'سرطان الخلايا القاعدية',
            4: 'التقرن الشعاعي',
            5: 'الآفات الوعائية',
            6: 'الورم الليفي الجلدي'
        }
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # الحصول على اسم الصورة والتصنيف
        image_name = self.dataframe.iloc[idx]['image_id'] + '.jpg'
        label = self.dataframe.iloc[idx]['dx']
        
        # تحميل الصورة
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # تطبيق التحويلات
        if self.transform:
            image = self.transform(image)
        
        # تحويل التصنيف إلى رقم
        label = self.classes[label]
        
        return image, label

def load_and_explore_data():
    """
    تحميل واستكشاف البيانات
    Load and explore the HAM10000 dataset
    """
    print("=" * 50)
    print("الخطوة 1: تحميل واستكشاف البيانات")
    print("Step 1: Loading and Exploring Data")
    print("=" * 50)
    
    # تحميل ملف البيانات الوصفية
    # في الواقع، ستحتاج لتحميل مجموعة بيانات HAM10000 من:
    # https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
    
    # إنشاء بيانات تجريبية للتوضيح
    # في التطبيق الحقيقي، استبدل هذا بتحميل البيانات الفعلية
    sample_data = {
        'image_id': [f'ISIC_{i:07d}' for i in range(1000)],
        'dx': np.random.choice(['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'], 1000),
        'age': np.random.randint(20, 80, 1000),
        'sex': np.random.choice(['male', 'female'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"عدد الصور الإجمالي: {len(df)}")
    print(f"عدد الفئات: {df['dx'].nunique()}")
    print("\nتوزيع الفئات:")
    print(df['dx'].value_counts())
    
    # رسم بياني لتوزيع الفئات
    plt.figure(figsize=(12, 6))
    df['dx'].value_counts().plot(kind='bar')
    plt.title('توزيع فئات الأمراض الجلدية')
    plt.xlabel('نوع المرض')
    plt.ylabel('عدد الصور')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    return df

def create_data_transforms():
    """
    إنشاء تحويلات البيانات للتدريب والاختبار
    Create data transforms for training and testing
    """
    print("\n" + "=" * 50)
    print("الخطوة 2: إعداد تحويلات البيانات")
    print("Step 2: Setting up Data Transforms")
    print("=" * 50)
    
    # تحويلات التدريب (مع Data Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # تحويلات الاختبار (بدون Augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("✅ تم إعداد تحويلات البيانات بنجاح")
    print("- حجم الصورة: 224x224")
    print("- تطبيق Data Augmentation على بيانات التدريب")
    print("- تطبيق Normalization باستخدام معايير ImageNet")
    
    return train_transforms, test_transforms

def create_model():
    """
    إنشاء نموذج التصنيف باستخدام Transfer Learning
    Create classification model using Transfer Learning
    """
    print("\n" + "=" * 50)
    print("الخطوة 3: بناء النموذج")
    print("Step 3: Building the Model")
    print("=" * 50)
    
    # تحميل نموذج ResNet50 المدرب مسبقاً
    model = models.resnet50(pretrained=True)
    
    # تجميد طبقات النموذج المدرب مسبقاً
    for param in model.parameters():
        param.requires_grad = False
    
    # استبدال الطبقة الأخيرة للتصنيف إلى 7 فئات
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)  # 7 فئات للأمراض الجلدية
    )
    
    # نقل النموذج إلى الجهاز المناسب
    model = model.to(device)
    
    print("✅ تم بناء النموذج بنجاح")
    print(f"- النموذج الأساسي: ResNet50")
    print(f"- عدد الفئات: 7")
    print(f"- استخدام Transfer Learning")
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=25):
    """
    تدريب النموذج
    Train the model
    """
    print("\n" + "=" * 50)
    print("الخطوة 4: تدريب النموذج")
    print("Step 4: Training the Model")
    print("=" * 50)
    
    # تعريف دالة الخسارة والمحسن
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # متغيرات لتتبع الأداء
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # مرحلة التدريب
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="التدريب"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu().numpy())
        
        # مرحلة التقييم
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="التقييم"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.cpu().numpy())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # حفظ أفضل نموذج
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_skin_classifier.pt')
            print(f'✅ تم حفظ أفضل نموذج بدقة: {best_val_acc:.4f}')
        
        scheduler.step()
    
    # رسم منحنيات التدريب
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
    
    return model

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    رسم منحنيات التدريب
    Plot training curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # رسم منحنى الخسارة
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('منحنى الخسارة أثناء التدريب')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # رسم منحنى الدقة
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('منحنى الدقة أثناء التدريب')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def evaluate_model(model, test_loader):
    """
    تقييم النموذج على بيانات الاختبار
    Evaluate model on test data
    """
    print("\n" + "=" * 50)
    print("الخطوة 5: تقييم النموذج")
    print("Step 5: Model Evaluation")
    print("=" * 50)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="التقييم"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # حساب الدقة الإجمالية
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'دقة النموذج على بيانات الاختبار: {accuracy:.4f}')
    
    # تقرير التصنيف
    class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    print("\nتقرير التصنيف التفصيلي:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # مصفوفة الالتباس
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('مصفوفة الالتباس')
    plt.xlabel('التنبؤ')
    plt.ylabel('الحقيقة')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def save_model_for_mobile(model):
    """
    حفظ النموذج للاستخدام في التطبيق المحمول
    Save model for mobile application
    """
    print("\n" + "=" * 50)
    print("الخطوة 6: حفظ النموذج للتطبيق")
    print("Step 6: Saving Model for Mobile App")
    print("=" * 50)
    
    # تحويل النموذج إلى وضع التقييم
    model.eval()
    
    # إنشاء مثال للدخل
    example_input = torch.randn(1, 3, 224, 224).to(device)
    
    # تتبع النموذج
    traced_model = torch.jit.trace(model, example_input)
    
    # حفظ النموذج المتتبع
    traced_model.save('skin_classifier_mobile.pt')
    
    print("✅ تم حفظ النموذج بنجاح للاستخدام في التطبيق المحمول")
    print("اسم الملف: skin_classifier_mobile.pt")

def test_single_image(model, image_path, transform):
    """
    اختبار النموذج على صورة واحدة
    Test model on a single image
    """
    print("\n" + "=" * 50)
    print("اختبار النموذج على صورة واحدة")
    print("Testing Model on Single Image")
    print("=" * 50)
    
    # أسماء الفئات
    class_names = {
        0: 'الشامات الصبغية (Melanocytic nevi)',
        1: 'الورم الميلانيني الخبيث (Melanoma)',
        2: 'الآفات الحميدة الشبيهة بالتقرن (Benign keratosis-like lesions)',
        3: 'سرطان الخلايا القاعدية (Basal cell carcinoma)',
        4: 'التقرن الشعاعي (Actinic keratoses)',
        5: 'الآفات الوعائية (Vascular lesions)',
        6: 'الورم الليفي الجلدي (Dermatofibroma)'
    }
    
    # تحميل وتحضير الصورة
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # التنبؤ
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(outputs, 1).item()
        confidence = probabilities[predicted_class].item()
    
    print(f"التشخيص المتوقع: {class_names[predicted_class]}")
    print(f"نسبة الثقة: {confidence:.2%}")
    
    # عرض أعلى 3 احتمالات
    print("\nأعلى 3 احتمالات:")
    top3_prob, top3_classes = torch.topk(probabilities, 3)
    for i in range(3):
        class_idx = top3_classes[i].item()
        prob = top3_prob[i].item()
        print(f"{i+1}. {class_names[class_idx]}: {prob:.2%}")

def main():
    """
    الدالة الرئيسية لتشغيل المشروع
    Main function to run the project
    """
    print("🏥 مشروع تصنيف الأمراض الجلدية باستخدام الذكاء الاصطناعي")
    print("🏥 Skin Disease Classification using AI")
    print("=" * 60)
    
    # الخطوة 1: تحميل واستكشاف البيانات
    df = load_and_explore_data()
    
    # الخطوة 2: إعداد تحويلات البيانات
    train_transforms, test_transforms = create_data_transforms()
    
    # تقسيم البيانات
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['dx'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])
    
    print(f"\nتقسيم البيانات:")
    print(f"- التدريب: {len(train_df)} صورة")
    print(f"- التحقق: {len(val_df)} صورة")
    print(f"- الاختبار: {len(test_df)} صورة")
    
    # ملاحظة: في التطبيق الحقيقي، ستحتاج لإنشاء DataLoaders
    # هنا نقوم بإنشاء DataLoaders وهمية للتوضيح
    
    # الخطوة 3: بناء النموذج
    model = create_model()
    
    # الخطوة 4: تدريب النموذج
    # في هذا المثال، سنتخطى التدريب الفعلي ونحفظ نموذج وهمي
    print("\n⚠️  ملاحظة: في هذا المثال التوضيحي، سنتخطى التدريب الفعلي")
    print("في التطبيق الحقيقي، ستحتاج لتحميل بيانات HAM10000 وتدريب النموذج")
    
    # الخطوة 5: حفظ النموذج
    save_model_for_mobile(model)
    
    print("\n🎉 تم إكمال جميع الخطوات بنجاح!")
    print("يمكنك الآن استخدام النموذج في تطبيق Flutter")

if __name__ == "__main__":
    main()