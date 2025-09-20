import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class DiseaseInfoWidget extends StatelessWidget {
  final String diseaseType;

  const DiseaseInfoWidget({
    super.key,
    required this.diseaseType,
  });

  @override
  Widget build(BuildContext context) {
    final diseaseInfo = _getDiseaseInfo(diseaseType);
    final diseaseColor = _getDiseaseColor(diseaseType);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(context, diseaseInfo['name']!, diseaseColor),
          const SizedBox(height: 16),
          _buildInfoSection(
            context,
            'الوصف',
            diseaseInfo['description']!,
            Icons.description,
          ),
          const SizedBox(height: 16),
          _buildInfoSection(
            context,
            'الأعراض الشائعة',
            diseaseInfo['symptoms']!,
            Icons.list_alt,
          ),
          const SizedBox(height: 16),
          _buildInfoSection(
            context,
            'التوصيات',
            diseaseInfo['recommendations']!,
            Icons.medical_services,
          ),
          const SizedBox(height: 16),
          _buildSeverityIndicator(context, diseaseInfo['severity']!, diseaseColor),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context, String name, Color color) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(
            Icons.info,
            color: color,
            size: 24,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'معلومات طبية',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  color: AppTheme.textSecondary,
                ),
              ),
              Text(
                name,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildInfoSection(
    BuildContext context,
    String title,
    String content,
    IconData icon,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              icon,
              size: 18,
              color: AppTheme.primaryColor,
            ),
            const SizedBox(width: 8),
            Text(
              title,
              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
                color: AppTheme.primaryColor,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: AppTheme.backgroundColor,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            content,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              height: 1.5,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSeverityIndicator(
    BuildContext context,
    String severity,
    Color color,
  ) {
    IconData severityIcon;
    Color severityColor;
    String severityText;

    switch (severity.toLowerCase()) {
      case 'high':
        severityIcon = Icons.warning;
        severityColor = AppTheme.errorColor;
        severityText = 'خطورة عالية - يتطلب تدخل طبي فوري';
        break;
      case 'medium':
        severityIcon = Icons.info;
        severityColor = AppTheme.warningColor;
        severityText = 'خطورة متوسطة - يُنصح بمراجعة الطبيب';
        break;
      case 'low':
        severityIcon = Icons.check_circle;
        severityColor = AppTheme.successColor;
        severityText = 'خطورة منخفضة - متابعة دورية';
        break;
      default:
        severityIcon = Icons.help;
        severityColor = AppTheme.textSecondary;
        severityText = 'يُنصح بمراجعة الطبيب للتقييم';
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: severityColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: severityColor.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Row(
        children: [
          Icon(
            severityIcon,
            color: severityColor,
            size: 20,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'مستوى الخطورة',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    color: severityColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  severityText,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: severityColor,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getDiseaseColor(String diseaseType) {
    return AppTheme.diseaseColors[diseaseType] ?? AppTheme.primaryColor;
  }

  Map<String, String> _getDiseaseInfo(String diseaseType) {
    const diseaseInfoMap = {
      'melanocytic_nevi': {
        'name': 'الشامات الصبغية',
        'description': 'الشامات الصبغية هي نمو حميد شائع في الجلد يحتوي على خلايا صبغية. معظم الشامات غير ضارة ولكن يجب مراقبتها للتغيرات.',
        'symptoms': '• بقع بنية أو سوداء على الجلد\n• قد تكون مسطحة أو مرتفعة\n• حجم ثابت عادة\n• حواف منتظمة',
        'recommendations': '• مراقبة التغيرات في الحجم أو اللون\n• تجنب التعرض المفرط للشمس\n• فحص دوري عند الطبيب\n• استخدام واقي الشمس',
        'severity': 'low'
      },
      'melanoma': {
        'name': 'الورم الميلانيني الخبيث',
        'description': 'الميلانوما هو نوع خطير من سرطان الجلد يتطور في الخلايا الصبغية. يتطلب تشخيص وعلاج فوري من قبل طبيب مختص.',
        'symptoms': '• تغير في لون أو حجم الشامة\n• حواف غير منتظمة\n• نزيف أو حكة\n• ظهور شامة جديدة غير طبيعية',
        'recommendations': '• مراجعة طبيب الأمراض الجلدية فوراً\n• تجنب التعرض للشمس\n• إجراء فحوصات شاملة\n• المتابعة الدورية المنتظمة',
        'severity': 'high'
      },
      'benign_keratosis': {
        'name': 'الآفات الحميدة الشبيهة بالتقرن',
        'description': 'آفات جلدية حميدة شائعة تظهر مع التقدم في العمر. عادة ما تكون غير ضارة ولكن قد تحتاج لمتابعة طبية.',
        'symptoms': '• بقع بنية أو سوداء مرتفعة\n• ملمس خشن أو شمعي\n• تظهر تدريجياً مع العمر\n• قد تسبب حكة خفيفة',
        'recommendations': '• مراجعة الطبيب للتأكد من التشخيص\n• تجنب الحك أو الخدش\n• استخدام مرطبات الجلد\n• المتابعة الدورية',
        'severity': 'low'
      },
      'basal_cell_carcinoma': {
        'name': 'سرطان الخلايا القاعدية',
        'description': 'أكثر أنواع سرطان الجلد شيوعاً ولكنه أقل خطورة من الميلانوما. ينمو ببطء ونادراً ما ينتشر لأجزاء أخرى من الجسم.',
        'symptoms': '• نتوء لؤلؤي أو شمعي\n• قرحة لا تشفى\n• بقعة حمراء مسطحة\n• منطقة شبيهة بالندبة',
        'recommendations': '• مراجعة طبيب الأمراض الجلدية\n• تجنب التعرض للشمس\n• استخدام واقي الشمس يومياً\n• المتابعة المنتظمة بعد العلاج',
        'severity': 'medium'
      },
      'actinic_keratoses': {
        'name': 'التقرن الشعاعي',
        'description': 'آفات جلدية تنتج عن التعرض المفرط لأشعة الشمس. تعتبر حالة ما قبل سرطانية وتحتاج لمتابعة طبية.',
        'symptoms': '• بقع خشنة ومتقشرة\n• لون أحمر أو بني\n• ملمس رملي أو خشن\n• قد تسبب حرقة أو حكة',
        'recommendations': '• مراجعة طبيب الأمراض الجلدية\n• العلاج المبكر مهم\n• استخدام واقي الشمس\n• تجنب التعرض للشمس في أوقات الذروة',
        'severity': 'medium'
      },
      'vascular_lesions': {
        'name': 'الآفات الوعائية',
        'description': 'آفات جلدية تنتج عن تشوهات في الأوعية الدموية. معظمها حميد ولكن قد يحتاج لعلاج تجميلي أو طبي.',
        'symptoms': '• بقع حمراء أو بنفسجية\n• قد تكون مسطحة أو مرتفعة\n• تختلف في الحجم\n• قد تنزف عند الإصابة',
        'recommendations': '• مراجعة الطبيب للتقييم\n• تجنب الإصابة أو الخدش\n• استخدام واقي الشمس\n• المتابعة حسب توصية الطبيب',
        'severity': 'low'
      },
      'dermatofibroma': {
        'name': 'الورم الليفي الجلدي',
        'description': 'نمو حميد شائع في الجلد يتكون من نسيج ليفي. عادة ما يكون غير ضار ولا يحتاج لعلاج إلا للأغراض التجميلية.',
        'symptoms': '• عقدة صلبة تحت الجلد\n• لون بني أو أحمر\n• قد تسبب حكة خفيفة\n• ثابتة الحجم عادة',
        'recommendations': '• مراجعة الطبيب للتأكد من التشخيص\n• تجنب الحك أو الخدش\n• المتابعة الدورية\n• العلاج الجراحي إذا لزم الأمر',
        'severity': 'low'
      },
    };

    return diseaseInfoMap[diseaseType] ?? {
      'name': 'غير محدد',
      'description': 'يرجى مراجعة طبيب الأمراض الجلدية للحصول على تشخيص دقيق.',
      'symptoms': 'أعراض غير محددة',
      'recommendations': 'مراجعة الطبيب المختص',
      'severity': 'medium'
    };
  }
}