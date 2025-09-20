"""
اختبار النموذج المدرب على صورة واحدة
Test the trained model on a single image
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

def load_model(model_path):
    """
    تحميل النموذج المحفوظ
    Load the saved model
    """
    model = torch.jit.load(model_path)
    model.eval()
    return model

def preprocess_image(image_path):
    """
    معالجة الصورة قبل التنبؤ
    Preprocess image before prediction
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_skin_disease(model, image_tensor):
    """
    التنبؤ بنوع المرض الجلدي
    Predict skin disease type
    """
    class_names = {
        0: 'الشامات الصبغية (Melanocytic nevi)',
        1: 'الورم الميلانيني الخبيث (Melanoma)',
        2: 'الآفات الحميدة الشبيهة بالتقرن (Benign keratosis-like lesions)',
        3: 'سرطان الخلايا القاعدية (Basal cell carcinoma)',
        4: 'التقرن الشعاعي (Actinic keratoses)',
        5: 'الآفات الوعائية (Vascular lesions)',
        6: 'الورم الليفي الجلدي (Dermatofibroma)'
    }
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(outputs, 1).item()
        confidence = probabilities[predicted_class].item()
    
    return {
        'predicted_class': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': probabilities.numpy()
    }

def main():
    """
    اختبار النموذج على صورة تجريبية
    Test model on a sample image
    """
    print("🧪 اختبار نموذج تصنيف الأمراض الجلدية")
    print("🧪 Testing Skin Disease Classification Model")
    print("=" * 50)
    
    # تحميل النموذج
    try:
        model = load_model('skin_classifier_mobile.pt')
        print("✅ تم تحميل النموذج بنجاح")
    except FileNotFoundError:
        print("❌ لم يتم العثور على ملف النموذج")
        print("يرجى تشغيل train_model.py أولاً")
        return
    
    # اختبار على صورة تجريبية
    # في التطبيق الحقيقي، استبدل هذا بمسار صورة حقيقية
    sample_image_path = "sample_skin_image.jpg"
    
    try:
        # معالجة الصورة
        image_tensor = preprocess_image(sample_image_path)
        
        # التنبؤ
        result = predict_skin_disease(model, image_tensor)
        
        print(f"\n📊 نتائج التشخيص:")
        print(f"التشخيص المتوقع: {result['class_name']}")
        print(f"نسبة الثقة: {result['confidence']:.2%}")
        
    except FileNotFoundError:
        print(f"❌ لم يتم العثور على الصورة: {sample_image_path}")
        print("يرجى وضع صورة تجريبية في نفس المجلد")

if __name__ == "__main__":
    main()