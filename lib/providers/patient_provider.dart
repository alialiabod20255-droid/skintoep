import 'package:flutter/foundation.dart';
import '../models/patient.dart';
import '../services/database_service.dart';

class PatientProvider with ChangeNotifier {
  List<Patient> _patients = [];
  bool _isLoading = false;
  String? _error;

  List<Patient> get patients => _patients;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> loadPatients() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _patients = await DatabaseService.instance.getAllPatients();
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> addPatient(Patient patient) async {
    try {
      final id = await DatabaseService.instance.insertPatient(patient);
      final newPatient = patient.copyWith(id: id);
      _patients.add(newPatient);
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  Future<void> updatePatient(Patient patient) async {
    try {
      await DatabaseService.instance.updatePatient(patient);
      final index = _patients.indexWhere((p) => p.id == patient.id);
      if (index != -1) {
        _patients[index] = patient;
        notifyListeners();
      }
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  Future<void> deletePatient(int id) async {
    try {
      await DatabaseService.instance.deletePatient(id);
      _patients.removeWhere((p) => p.id == id);
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  Patient? getPatientById(int id) {
    try {
      return _patients.firstWhere((p) => p.id == id);
    } catch (e) {
      return null;
    }
  }

  List<Patient> searchPatients(String query) {
    if (query.isEmpty) return _patients;
    
    return _patients.where((patient) {
      return patient.name.toLowerCase().contains(query.toLowerCase()) ||
             patient.phone.contains(query) ||
             patient.nationalId.contains(query);
    }).toList();
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }
}