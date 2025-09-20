import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsProvider with ChangeNotifier {
  bool _isDarkMode = false;
  String _language = 'ar';
  bool _notificationsEnabled = true;
  bool _soundEnabled = true;
  double _confidenceThreshold = 0.7;
  bool _autoSaveResults = true;
  String _exportFormat = 'pdf';

  bool get isDarkMode => _isDarkMode;
  String get language => _language;
  bool get notificationsEnabled => _notificationsEnabled;
  bool get soundEnabled => _soundEnabled;
  double get confidenceThreshold => _confidenceThreshold;
  bool get autoSaveResults => _autoSaveResults;
  String get exportFormat => _exportFormat;

  SettingsProvider() {
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    
    _isDarkMode = prefs.getBool('isDarkMode') ?? false;
    _language = prefs.getString('language') ?? 'ar';
    _notificationsEnabled = prefs.getBool('notificationsEnabled') ?? true;
    _soundEnabled = prefs.getBool('soundEnabled') ?? true;
    _confidenceThreshold = prefs.getDouble('confidenceThreshold') ?? 0.7;
    _autoSaveResults = prefs.getBool('autoSaveResults') ?? true;
    _exportFormat = prefs.getString('exportFormat') ?? 'pdf';
    
    notifyListeners();
  }

  Future<void> setDarkMode(bool value) async {
    _isDarkMode = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isDarkMode', value);
    notifyListeners();
  }

  Future<void> setLanguage(String value) async {
    _language = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('language', value);
    notifyListeners();
  }

  Future<void> setNotificationsEnabled(bool value) async {
    _notificationsEnabled = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('notificationsEnabled', value);
    notifyListeners();
  }

  Future<void> setSoundEnabled(bool value) async {
    _soundEnabled = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('soundEnabled', value);
    notifyListeners();
  }

  Future<void> setConfidenceThreshold(double value) async {
    _confidenceThreshold = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble('confidenceThreshold', value);
    notifyListeners();
  }

  Future<void> setAutoSaveResults(bool value) async {
    _autoSaveResults = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('autoSaveResults', value);
    notifyListeners();
  }

  Future<void> setExportFormat(String value) async {
    _exportFormat = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('exportFormat', value);
    notifyListeners();
  }

  Future<void> resetSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
    
    _isDarkMode = false;
    _language = 'ar';
    _notificationsEnabled = true;
    _soundEnabled = true;
    _confidenceThreshold = 0.7;
    _autoSaveResults = true;
    _exportFormat = 'pdf';
    
    notifyListeners();
  }
}