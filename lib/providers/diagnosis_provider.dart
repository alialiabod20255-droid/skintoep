import 'package:flutter/foundation.dart';
import '../models/diagnosis.dart';
import '../services/database_service.dart';
import '../services/ai_service.dart';
import 'dart:io';

class DiagnosisProvider with ChangeNotifier {
  List<Diagnosis> _diagnoses = [];
  bool _isLoading = false;
  bool _isAnalyzing = false;
  String? _error;
  Diagnosis? _currentDiagnosis;

  List<Diagnosis> get diagnoses => _diagnoses;
  bool get isLoading => _isLoading;
  bool get isAnalyzing => _isAnalyzing;
  String? get error => _error;
  Diagnosis? get currentDiagnosis => _currentDiagnosis;

  final AIService _aiService = AIService();

  Future<void> loadRecentDiagnoses({int limit = 10}) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _diagnoses = await DatabaseService.instance.getRecentDiagnoses(limit);
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loadPatientDiagnoses(int patientId) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _diagnoses = await DatabaseService.instance.getPatientDiagnoses(patientId);
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<Diagnosis?> analyzeImage(File imageFile, {int? patientId}) async {
    _isAnalyzing = true;
    _error = null;
    _currentDiagnosis = null;
    notifyListeners();

    try {
      // تحليل الصورة باستخدام الذكاء الاصطناعي
      final result = await _aiService.analyzeImage(imageFile);
      
      // إنشاء تشخيص جديد
      final diagnosis = Diagnosis(
        patientId: patientId,
        imagePath: imageFile.path,
        predictedClass: result.predictedClass,
        className: result.className,
        confidence: result.confidence,
        allProbabilities: result.allProbabilities,
        createdAt: DateTime.now(),
        notes: '',
      );

      // حفظ التشخيص في قاعدة البيانات
      final id = await DatabaseService.instance.insertDiagnosis(diagnosis);
      _currentDiagnosis = diagnosis.copyWith(id: id);
      
      // إضافة التشخيص إلى القائمة
      _diagnoses.insert(0, _currentDiagnosis!);
      
      notifyListeners();
      return _currentDiagnosis;
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return null;
    } finally {
      _isAnalyzing = false;
      notifyListeners();
    }
  }

  Future<void> updateDiagnosisNotes(int diagnosisId, String notes) async {
    try {
      await DatabaseService.instance.updateDiagnosisNotes(diagnosisId, notes);
      
      // تحديث التشخيص في القائمة
      final index = _diagnoses.indexWhere((d) => d.id == diagnosisId);
      if (index != -1) {
        _diagnoses[index] = _diagnoses[index].copyWith(notes: notes);
        notifyListeners();
      }
      
      // تحديث التشخيص الحالي إذا كان هو نفسه
      if (_currentDiagnosis?.id == diagnosisId) {
        _currentDiagnosis = _currentDiagnosis!.copyWith(notes: notes);
        notifyListeners();
      }
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  Future<void> deleteDiagnosis(int id) async {
    try {
      await DatabaseService.instance.deleteDiagnosis(id);
      _diagnoses.removeWhere((d) => d.id == id);
      
      if (_currentDiagnosis?.id == id) {
        _currentDiagnosis = null;
      }
      
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  void setCurrentDiagnosis(Diagnosis? diagnosis) {
    _currentDiagnosis = diagnosis;
    notifyListeners();
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }

  void clearCurrentDiagnosis() {
    _currentDiagnosis = null;
    notifyListeners();
  }

  // إحصائيات التشخيص
  Map<String, int> getDiagnosisStatistics() {
    final stats = <String, int>{};
    
    for (final diagnosis in _diagnoses) {
      final className = diagnosis.className;
      stats[className] = (stats[className] ?? 0) + 1;
    }
    
    return stats;
  }

  // الحصول على أحدث التشخيصات لمريض معين
  List<Diagnosis> getPatientRecentDiagnoses(int patientId, {int limit = 5}) {
    return _diagnoses
        .where((d) => d.patientId == patientId)
        .take(limit)
        .toList();
  }
}