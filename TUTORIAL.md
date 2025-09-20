# دليل شامل: مشروع تصنيف الأمراض الجلدية باستخدام الذكاء الاصطناعي

## 🎯 نظرة عامة على المشروع

هذا المشروع يجمع بين قوة الذكاء الاصطناعي وسهولة استخدام تطبيقات الهاتف المحمول لإنشاء نظام تشخيص أولي للأمراض الجلدية. يتكون المشروع من جزأين رئيسيين:

1. **نموذج الذكاء الاصطناعي (Python + PyTorch)**: لتدريب نموذج تصنيف الصور
2. **تطبيق الهاتف المحمول (Flutter)**: لاستخدام النموذج في الواقع

---

## 📋 المتطلبات الأساسية

### للجزء الخاص بـ Python:
```bash
Python 3.8+
PyTorch 1.9+
torchvision
pandas
numpy
matplotlib
scikit-learn
Pillow
```

### للجزء الخاص بـ Flutter:
```bash
Flutter SDK 3.0+
Dart 2.17+
Android Studio أو VS Code
```

---

## 🧠 الجزء الأول: بناء نموذج الذكاء الاصطناعي

### الخطوة 1: إعداد البيئة

```bash
# إنشاء بيئة افتراضية
python -m venv skin_disease_env

# تفعيل البيئة الافتراضية
# على Windows:
skin_disease_env\Scripts\activate
# على macOS/Linux:
source skin_disease_env/bin/activate

# تثبيت المكتبات المطلوبة
pip install -r python/requirements.txt
```

### الخطوة 2: تحميل مجموعة البيانات

مجموعة بيانات HAM10000 تحتوي على أكثر من 10,000 صورة للأمراض الجلدية مقسمة إلى 7 فئات:

1. **nv** - Melanocytic nevi (الشامات الصبغية)
2. **mel** - Melanoma (الورم الميلانيني الخبيث)
3. **bkl** - Benign keratosis-like lesions (الآفات الحميدة الشبيهة بالتقرن)
4. **bcc** - Basal cell carcinoma (سرطان الخلايا القاعدية)
5. **akiec** - Actinic keratoses (التقرن الشعاعي)
6. **vasc** - Vascular lesions (الآفات الوعائية)
7. **df** - Dermatofibroma (الورم الليفي الجلدي)

```python
# تحميل البيانات من Kaggle
# https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

import pandas as pd
import numpy as np

# قراءة ملف البيانات الوصفية
metadata = pd.read_csv('HAM10000_metadata.csv')
print(f"عدد الصور: {len(metadata)}")
print(f"توزيع الفئات:\n{metadata['dx'].value_counts()}")
```

### الخطوة 3: استكشاف البيانات

```python
import matplotlib.pyplot as plt
import seaborn as sns

# رسم توزيع الفئات
plt.figure(figsize=(12, 6))
metadata['dx'].value_counts().plot(kind='bar')
plt.title('توزيع فئات الأمراض الجلدية')
plt.xlabel('نوع المرض')
plt.ylabel('عدد الصور')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# إحصائيات إضافية
print(f"متوسط العمر: {metadata['age'].mean():.1f}")
print(f"توزيع الجنس:\n{metadata['sex'].value_counts()}")
```

### الخطوة 4: معالجة البيانات المسبقة

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# تحويلات التدريب مع Data Augmentation
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

# تحويلات الاختبار
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# تقسيم البيانات
train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

print(f"التدريب: {len(train_df)} صورة")
print(f"التحقق: {len(val_df)} صورة")
print(f"الاختبار: {len(test_df)} صورة")
```

### الخطوة 5: بناء النموذج

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinDiseaseClassifier, self).__init__()
        
        # تحميل ResNet50 المدرب مسبقاً
        self.backbone = models.resnet50(pretrained=True)
        
        # تجميد طبقات النموذج الأساسي
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # استبدال الطبقة الأخيرة
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# إنشاء النموذج
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkinDiseaseClassifier(num_classes=7).to(device)

print(f"النموذج يعمل على: {device}")
print(f"عدد المعاملات القابلة للتدريب: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

### الخطوة 6: تدريب النموذج

```python
import torch.optim as optim
from tqdm import tqdm

# إعداد التدريب
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.backbone.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# دالة التدريب
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in tqdm(dataloader, desc="التدريب"):
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
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

# دالة التقييم
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="التقييم"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

# حلقة التدريب الرئيسية
num_epochs = 25
best_val_acc = 0.0
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 30)
    
    # التدريب
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc.cpu().numpy())
    
    # التقييم
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc.cpu().numpy())
    
    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    # حفظ أفضل نموذج
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_skin_classifier.pt')
        print(f'✅ تم حفظ أفضل نموذج بدقة: {best_val_acc:.4f}')
    
    scheduler.step()
```

### الخطوة 7: تقييم النموذج

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="التقييم النهائي"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # حساب الدقة
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'دقة النموذج على بيانات الاختبار: {accuracy:.4f}')
    
    # تقرير التصنيف
    class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    print("\nتقرير التصنيف:")
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
    plt.show()

# تقييم النموذج
evaluate_model(model, test_loader, device)
```

### الخطوة 8: حفظ النموذج للتطبيق المحمول

```python
def save_model_for_mobile(model, save_path='skin_classifier_mobile.pt'):
    """حفظ النموذج بتنسيق TorchScript للاستخدام في التطبيق المحمول"""
    
    # تحويل النموذج إلى وضع التقييم
    model.eval()
    
    # إنشاء مثال للدخل
    example_input = torch.randn(1, 3, 224, 224).to(device)
    
    # تتبع النموذج
    traced_model = torch.jit.trace(model, example_input)
    
    # حفظ النموذج المتتبع
    traced_model.save(save_path)
    
    print(f"✅ تم حفظ النموذج بنجاح: {save_path}")
    print(f"حجم الملف: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

# حفظ النموذج
save_model_for_mobile(model)
```

---

## 📱 الجزء الثاني: تطبيق Flutter

### الخطوة 1: إعداد مشروع Flutter

```bash
# إنشاء مشروع جديد
flutter create skin_diseases
cd skin_diseases

# إضافة المكتبات المطلوبة
flutter pub add image_picker
flutter pub add permission_handler
flutter pub add provider
flutter pub add shared_preferences
flutter pub add http
flutter pub add path_provider
```

### الخطوة 2: هيكل المشروع

```
lib/
├── main.dart                 # نقطة البداية
├── screens/
│   └── home_screen.dart     # الشاشة الرئيسية
├── widgets/
│   ├── custom_app_bar.dart  # شريط التطبيق المخصص
│   ├── image_picker_widget.dart  # واجهة اختيار الصور
│   ├── diagnosis_result_widget.dart  # عرض النتائج
│   └── disease_info_widget.dart     # معلومات المرض
├── services/
│   └── ai_service.dart      # خدمة الذكاء الاصطناعي
├── models/
│   └── diagnosis_result.dart # نموذج البيانات
└── utils/
    └── app_theme.dart       # تصميم التطبيق
```

### الخطوة 3: تصميم واجهة المستخدم

التطبيق يتميز بتصميم طبي عصري واحترافي مع:

- **ألوان طبية هادئة**: أزرق وأخضر مع لمسات من الأبيض
- **خطوط عربية جميلة**: خط Cairo للنصوص العربية
- **رسوم متحركة سلسة**: انتقالات ناعمة بين الشاشات
- **تصميم متجاوب**: يعمل على جميع أحجام الشاشات

### الخطوة 4: دمج نموذج الذكاء الاصطناعي

```dart
// في الملف lib/services/ai_service.dart

import 'package:pytorch_mobile/pytorch_mobile.dart';

class AIService {
  Model? _model;
  
  Future<void> loadModel() async {
    try {
      _model = await PytorchMobile.loadModel('assets/models/skin_classifier_mobile.pt');
    } catch (e) {
      print('خطأ في تحميل النموذج: $e');
    }
  }
  
  Future<DiagnosisResult> analyzeImage(File imageFile) async {
    if (_model == null) {
      await loadModel();
    }
    
    // معالجة الصورة
    final imageBytes = await imageFile.readAsBytes();
    
    // التنبؤ
    final prediction = await _model!.predict(imageBytes);
    
    // معالجة النتائج وإرجاعها
    return _processResults(prediction);
  }
}
```

### الخطوة 5: إضافة النموذج إلى التطبيق

1. انسخ ملف `skin_classifier_mobile.pt` إلى مجلد `assets/models/`
2. أضف المسار في `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/skin_classifier_mobile.pt
    - assets/images/
```

---

## 🚀 تشغيل المشروع

### تدريب النموذج:

```bash
cd python
python train_model.py
```

### تشغيل التطبيق:

```bash
flutter pub get
flutter run
```

---

## 📊 النتائج المتوقعة

### أداء النموذج:
- **دقة التدريب**: ~95%
- **دقة التحقق**: ~88%
- **دقة الاختبار**: ~85%

### الفئات المدعومة:
1. الشامات الصبغية (دقة: ~90%)
2. الورم الميلانيني الخبيث (دقة: ~85%)
3. الآفات الحميدة (دقة: ~88%)
4. سرطان الخلايا القاعدية (دقة: ~82%)
5. التقرن الشعاعي (دقة: ~80%)
6. الآفات الوعائية (دقة: ~87%)
7. الورم الليفي الجلدي (دقة: ~89%)

---

## ⚠️ تنبيهات مهمة

### للاستخدام الطبي:
- هذا النظام للأغراض التعليمية فقط
- لا يغني عن استشارة الطبيب المختص
- النتائج قد تحتوي على أخطاء
- يجب التحقق من التشخيص طبياً

### للتطوير:
- تأكد من جودة البيانات المستخدمة
- اختبر النموذج على بيانات متنوعة
- راقب الأداء باستمرار
- حدث النموذج دورياً

---

## 🔧 استكشاف الأخطاء

### مشاكل شائعة:

1. **خطأ في تحميل النموذج**:
   - تأكد من وجود الملف في المسار الصحيح
   - تحقق من صيغة الملف (.pt)

2. **بطء في التحليل**:
   - قلل حجم الصورة
   - استخدم ضغط الصور

3. **دقة منخفضة**:
   - زد عدد epochs في التدريب
   - استخدم data augmentation أكثر
   - جرب نماذج أخرى

---

## 📚 مصادر إضافية

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flutter Documentation](https://flutter.dev/docs)
- [HAM10000 Dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## 🎯 الخطوات التالية

1. **تحسين النموذج**:
   - جرب نماذج أخرى (EfficientNet, DenseNet)
   - استخدم ensemble methods
   - طبق تقنيات regularization متقدمة

2. **تطوير التطبيق**:
   - أضف ميزات إضافية (تاريخ المريض، تتبع التغيرات)
   - حسن واجهة المستخدم
   - أضف دعم لغات متعددة

3. **النشر**:
   - اختبر على أجهزة مختلفة
   - حسن الأداء
   - انشر على متاجر التطبيقات

---

هذا المشروع يوضح كيفية دمج الذكاء الاصطناعي مع تطبيقات الهاتف المحمول لإنشاء حلول طبية مفيدة. تذكر أن الهدف هو التعلم والتطوير، وليس الاستخدام الطبي الفعلي.