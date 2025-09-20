# مشروع تصنيف الأمراض الجلدية باستخدام الذكاء الاصطناعي
## Skin Disease Classification using AI - PyTorch & Flutter

### نظرة عامة على المشروع

هذا المشروع يهدف إلى بناء نظام ذكي لتصنيف الأمراض الجلدية باستخدام تقنيات التعلم العميق. يتكون المشروع من جزأين رئيسيين:

1. **نموذج الذكاء الاصطناعي (Python - PyTorch)**: لتدريب نموذج تصنيف الصور
2. **تطبيق الهاتف المحمول (Flutter)**: لاستخدام النموذج في الواقع

### الأمراض الجلدية المدعومة (7 فئات)

1. **Melanocytic nevi** - الشامات الصبغية
2. **Melanoma** - الورم الميلانيني الخبيث
3. **Benign keratosis-like lesions** - الآفات الحميدة الشبيهة بالتقرن
4. **Basal cell carcinoma** - سرطان الخلايا القاعدية
5. **Actinic keratoses** - التقرن الشعاعي
6. **Vascular lesions** - الآفات الوعائية
7. **Dermatofibroma** - الورم الليفي الجلدي

### متطلبات المشروع

#### للجزء الخاص بـ Python:
- Python 3.8+
- PyTorch 1.9+
- torchvision
- pandas
- numpy
- matplotlib
- scikit-learn
- Pillow

#### للجزء الخاص بـ Flutter:
- Flutter SDK 3.0+
- Dart 2.17+
- Android Studio / VS Code
- مكتبات Flutter المطلوبة (موضحة في pubspec.yaml)

### هيكل المشروع

```
skin_diseases/
├── lib/                    # كود Flutter
├── python/                 # كود Python للتدريب
├── assets/                 # الأصول والنماذج
├── android/               # إعدادات Android
├── ios/                   # إعدادات iOS
└── README.md              # هذا الملف
```

### كيفية تشغيل المشروع

1. **تدريب النموذج**:
   ```bash
   cd python
   python train_model.py
   ```

2. **تشغيل التطبيق**:
   ```bash
   flutter pub get
   flutter run
   ```

### الميزات الرئيسية

- ✅ تصنيف دقيق للأمراض الجلدية
- ✅ واجهة مستخدم طبية احترافية
- ✅ دعم الكاميرا والمعرض
- ✅ عرض نسبة الثقة في التشخيص
- ✅ تصميم متجاوب لجميع الأجهزة
- ✅ معلومات تفصيلية عن كل مرض

### المراجع والمصادر

- HAM10000 Dataset
- PyTorch Documentation
- Flutter Documentation
- Project Gurukul Methodology