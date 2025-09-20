import 'package:flutter/material.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';
import 'package:provider/provider.dart';
import '../../providers/patient_provider.dart';
import '../../providers/diagnosis_provider.dart';
import '../../utils/app_theme.dart';
import '../../widgets/custom_app_bar.dart';
import '../../widgets/dashboard_card.dart';
import '../../widgets/recent_diagnoses_widget.dart';
import '../../widgets/statistics_chart.dart';
import '../patients/add_patient_screen.dart';
import '../diagnosis/diagnosis_screen.dart';
import '../reports/reports_screen.dart';
import '../settings/settings_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with TickerProviderStateMixin {
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _animationController.forward();
    
    // تحديث البيانات
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<PatientProvider>(context, listen: false).loadPatients();
      Provider.of<DiagnosisProvider>(context, listen: false).loadRecentDiagnoses();
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundColor,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _refreshData,
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const CustomAppBar(
                  title: 'لوحة التحكم',
                  showBackButton: false,
                ),
                Padding(
                  padding: const EdgeInsets.all(20),
                  child: AnimationLimiter(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: AnimationConfiguration.toStaggeredList(
                        duration: const Duration(milliseconds: 600),
                        childAnimationBuilder: (widget) => SlideAnimation(
                          horizontalOffset: 50.0,
                          child: FadeInAnimation(child: widget),
                        ),
                        children: [
                          _buildWelcomeSection(),
                          const SizedBox(height: 24),
                          _buildQuickActionsGrid(),
                          const SizedBox(height: 24),
                          _buildStatisticsSection(),
                          const SizedBox(height: 24),
                          const RecentDiagnosesWidget(),
                          const SizedBox(height: 24),
                          _buildTodaySchedule(),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _refreshData() async {
    await Future.wait([
      Provider.of<PatientProvider>(context, listen: false).loadPatients(),
      Provider.of<DiagnosisProvider>(context, listen: false).loadRecentDiagnoses(),
    ]);
  }

  Widget _buildWelcomeSection() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: AppTheme.primaryGradient,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppTheme.primaryColor.withOpacity(0.3),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'مرحباً بك',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'نظام تشخيص الأمراض الجلدية المتقدم',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.white.withOpacity(0.9),
                  ),
                ),
                const SizedBox(height: 16),
                Consumer<PatientProvider>(
                  builder: (context, patientProvider, child) {
                    return Text(
                      'إجمالي المرضى: ${patientProvider.patients.length}',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Icon(
              Icons.medical_services,
              size: 40,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickActionsGrid() {
    final actions = [
      {
        'icon': Icons.person_add,
        'title': 'مريض جديد',
        'subtitle': 'إضافة مريض جديد',
        'color': AppTheme.primaryColor,
        'onTap': () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const AddPatientScreen()),
        ),
      },
      {
        'icon': Icons.camera_alt,
        'title': 'تشخيص جديد',
        'subtitle': 'فحص صورة جديدة',
        'color': AppTheme.successColor,
        'onTap': () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const DiagnosisScreen()),
        ),
      },
      {
        'icon': Icons.assessment,
        'title': 'التقارير',
        'subtitle': 'عرض التقارير',
        'color': AppTheme.warningColor,
        'onTap': () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const ReportsScreen()),
        ),
      },
      {
        'icon': Icons.settings,
        'title': 'الإعدادات',
        'subtitle': 'إعدادات التطبيق',
        'color': AppTheme.textSecondary,
        'onTap': () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const SettingsScreen()),
        ),
      },
    ];

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 16,
        mainAxisSpacing: 16,
        childAspectRatio: 1.2,
      ),
      itemCount: actions.length,
      itemBuilder: (context, index) {
        final action = actions[index];
        return DashboardCard(
          icon: action['icon'] as IconData,
          title: action['title'] as String,
          subtitle: action['subtitle'] as String,
          color: action['color'] as Color,
          onTap: action['onTap'] as VoidCallback,
        );
      },
    );
  }

  Widget _buildStatisticsSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'الإحصائيات',
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        const StatisticsChart(),
      ],
    );
  }

  Widget _buildTodaySchedule() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'جدول اليوم',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            TextButton(
              onPressed: () {
                // الانتقال إلى صفحة الجدول الكامل
              },
              child: const Text('عرض الكل'),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Container(
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
            children: [
              _buildScheduleItem(
                time: '09:00',
                title: 'فحص المريض أحمد محمد',
                type: 'استشارة',
                color: AppTheme.primaryColor,
              ),
              const Divider(),
              _buildScheduleItem(
                time: '10:30',
                title: 'متابعة المريضة فاطمة علي',
                type: 'متابعة',
                color: AppTheme.successColor,
              ),
              const Divider(),
              _buildScheduleItem(
                time: '14:00',
                title: 'فحص جديد - سارة أحمد',
                type: 'فحص أولي',
                color: AppTheme.warningColor,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildScheduleItem({
    required String time,
    required String title,
    required String type,
    required Color color,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Container(
            width: 60,
            height: 40,
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Center(
              child: Text(
                time,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    type,
                    style: TextStyle(
                      color: color,
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}