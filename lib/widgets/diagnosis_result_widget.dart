import 'package:flutter/material.dart';
import '../models/diagnosis_result.dart';
import '../utils/app_theme.dart';

class DiagnosisResultWidget extends StatefulWidget {
  final DiagnosisResult result;

  const DiagnosisResultWidget({
    super.key,
    required this.result,
  });

  @override
  State<DiagnosisResultWidget> createState() => _DiagnosisResultWidgetState();
}

class _DiagnosisResultWidgetState extends State<DiagnosisResultWidget>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _progressAnimation;

  @override
  void initState() {
    super.initState();
    _setupAnimations();
  }

  void _setupAnimations() {
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(
      begin: 0.8,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: const Interval(0.0, 0.5, curve: Curves.elasticOut),
    ));

    _progressAnimation = Tween<double>(
      begin: 0.0,
      end: widget.result.confidence,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: const Interval(0.3, 1.0, curve: Curves.easeOutCubic),
    ));

    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _scaleAnimation,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 15,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            const SizedBox(height: 20),
            _buildMainResult(),
            const SizedBox(height: 20),
            _buildConfidenceIndicator(),
            const SizedBox(height: 20),
            _buildTopPredictions(),
            const SizedBox(height: 16),
            _buildDisclaimer(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: AppTheme.primaryColor.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: const Icon(
            Icons.analytics,
            color: AppTheme.primaryColor,
            size: 24,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'نتائج التحليل',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'تم التحليل باستخدام الذكاء الاصطناعي',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMainResult() {
    final diseaseColor = _getDiseaseColor(widget.result.predictedClass);
    
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: diseaseColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: diseaseColor.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.medical_information,
                color: diseaseColor,
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                'التشخيص المتوقع',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  color: diseaseColor,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            widget.result.className,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.bold,
              color: AppTheme.textPrimary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfidenceIndicator() {
    final confidence = widget.result.confidence;
    final color = _getConfidenceColor(confidence);
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'مستوى الثقة',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            AnimatedBuilder(
              animation: _progressAnimation,
              builder: (context, child) {
                return Text(
                  '${(_progressAnimation.value * 100).toInt()}%',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
                );
              },
            ),
          ],
        ),
        const SizedBox(height: 8),
        AnimatedBuilder(
          animation: _progressAnimation,
          builder: (context, child) {
            return LinearProgressIndicator(
              value: _progressAnimation.value,
              backgroundColor: color.withOpacity(0.2),
              valueColor: AlwaysStoppedAnimation<Color>(color),
              minHeight: 8,
            );
          },
        ),
        const SizedBox(height: 8),
        Text(
          _getConfidenceText(confidence),
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: AppTheme.textSecondary,
          ),
        ),
      ],
    );
  }

  Widget _buildTopPredictions() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'احتمالات أخرى',
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: 12),
        ...widget.result.allProbabilities.entries
            .where((entry) => entry.key != widget.result.predictedClass)
            .take(3)
            .map((entry) => _buildPredictionItem(entry.key, entry.value))
            .toList(),
      ],
    );
  }

  Widget _buildPredictionItem(String diseaseType, double probability) {
    final diseaseColor = _getDiseaseColor(diseaseType);
    
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: diseaseColor,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _getDiseaseNameInArabic(diseaseType),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Text(
            '${(probability * 100).toInt()}%',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w500,
              color: diseaseColor,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.warningColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: AppTheme.warningColor.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.warning_amber,
            color: AppTheme.warningColor,
            size: 16,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'هذا التشخيص أولي ولا يغني عن استشارة الطبيب المختص',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.warningColor,
                height: 1.3,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _getDiseaseColor(String diseaseType) {
    return AppTheme.diseaseColors[diseaseType] ?? AppTheme.primaryColor;
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return AppTheme.successColor;
    if (confidence >= 0.6) return AppTheme.warningColor;
    return AppTheme.errorColor;
  }

  String _getConfidenceText(double confidence) {
    if (confidence >= 0.8) return 'ثقة عالية في التشخيص';
    if (confidence >= 0.6) return 'ثقة متوسطة في التشخيص';
    return 'ثقة منخفضة - يُنصح بإعادة التصوير';
  }

  String _getDiseaseNameInArabic(String diseaseType) {
    const diseaseNames = {
      'melanocytic_nevi': 'الشامات الصبغية',
      'melanoma': 'الورم الميلانيني الخبيث',
      'benign_keratosis': 'الآفات الحميدة الشبيهة بالتقرن',
      'basal_cell_carcinoma': 'سرطان الخلايا القاعدية',
      'actinic_keratoses': 'التقرن الشعاعي',
      'vascular_lesions': 'الآفات الوعائية',
      'dermatofibroma': 'الورم الليفي الجلدي',
    };
    
    return diseaseNames[diseaseType] ?? diseaseType;
  }
}