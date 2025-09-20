import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/diagnosis_provider.dart';
import '../utils/app_theme.dart';

class StatisticsChart extends StatelessWidget {
  const StatisticsChart({super.key});

  @override
  Widget build(BuildContext context) {
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
          Text(
            'إحصائيات التشخيص',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 20),
          Consumer<DiagnosisProvider>(
            builder: (context, diagnosisProvider, child) {
              final stats = diagnosisProvider.getDiagnosisStatistics();
              
              if (stats.isEmpty) {
                return _buildEmptyStats();
              }

              return Column(
                children: stats.entries.map((entry) {
                  final total = stats.values.fold(0, (sum, count) => sum + count);
                  final percentage = (entry.value / total * 100).toInt();
                  
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 16),
                    child: _buildStatItem(
                      label: _getDiseaseNameInArabic(entry.key),
                      count: entry.value,
                      percentage: percentage,
                      color: _getDiseaseColor(entry.key),
                    ),
                  );
                }).toList(),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyStats() {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          Icon(
            Icons.bar_chart,
            size: 50,
            color: AppTheme.textLight,
          ),
          const SizedBox(height: 16),
          Text(
            'لا توجد إحصائيات متاحة',
            style: TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem({
    required String label,
    required int count,
    required int percentage,
    required Color color,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            Text(
              '$count ($percentage%)',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        LinearProgressIndicator(
          value: percentage / 100,
          backgroundColor: color.withOpacity(0.2),
          valueColor: AlwaysStoppedAnimation<Color>(color),
          minHeight: 6,
        ),
      ],
    );
  }

  Color _getDiseaseColor(String diseaseType) {
    const colors = {
      'melanocytic_nevi': Color(0xFF3498DB),
      'melanoma': Color(0xFFE74C3C),
      'benign_keratosis': Color(0xFF27AE60),
      'basal_cell_carcinoma': Color(0xFFE67E22),
      'actinic_keratoses': Color(0xFFF39C12),
      'vascular_lesions': Color(0xFFE91E63),
      'dermatofibroma': Color(0xFF9B59B6),
    };
    return colors[diseaseType] ?? AppTheme.primaryColor;
  }

  String _getDiseaseNameInArabic(String diseaseType) {
    const names = {
      'melanocytic_nevi': 'الشامات الصبغية',
      'melanoma': 'الورم الميلانيني الخبيث',
      'benign_keratosis': 'الآفات الحميدة الشبيهة بالتقرن',
      'basal_cell_carcinoma': 'سرطان الخلايا القاعدية',
      'actinic_keratoses': 'التقرن الشعاعي',
      'vascular_lesions': 'الآفات الوعائية',
      'dermatofibroma': 'الورم الليفي الجلدي',
    };
    return names[diseaseType] ?? diseaseType;
  }
}