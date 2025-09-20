import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'screens/splash_screen.dart';
import 'providers/patient_provider.dart';
import 'providers/diagnosis_provider.dart';
import 'providers/settings_provider.dart';
import 'providers/auth_provider.dart';
import 'utils/app_theme.dart';
import 'services/database_service.dart';
import 'services/notification_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // تهيئة قاعدة البيانات
  await DatabaseService.instance.database;
  
  // تهيئة الإشعارات
  await NotificationService.initialize();
  
  // إعداد شريط الحالة
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );

  runApp(const SkinDiseaseApp());
}

class SkinDiseaseApp extends StatelessWidget {
  const SkinDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => PatientProvider()),
        ChangeNotifierProvider(create: (_) => DiagnosisProvider()),
        ChangeNotifierProvider(create: (_) => SettingsProvider()),
      ],
      child: Consumer<SettingsProvider>(
        builder: (context, settingsProvider, child) {
          return MaterialApp(
            title: 'تشخيص الأمراض الجلدية',
            debugShowCheckedModeBanner: false,
            theme: settingsProvider.isDarkMode 
                ? AppTheme.darkTheme 
                : AppTheme.lightTheme,
            home: const SplashScreen(),
            locale: Locale(settingsProvider.language, ''),
            supportedLocales: const [
              Locale('ar', ''),
              Locale('en', ''),
            ],
          );
        },
      ),
    );
  }
}