import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user.dart';

class AuthProvider with ChangeNotifier {
  User? _currentUser;
  bool _isLoading = false;
  String? _error;

  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get isLoggedIn => _currentUser != null;

  Future<bool> checkLoginStatus() async {
    final prefs = await SharedPreferences.getInstance();
    final userId = prefs.getInt('userId');
    final userName = prefs.getString('userName');
    final userEmail = prefs.getString('userEmail');
    final userRole = prefs.getString('userRole');

    if (userId != null && userName != null && userEmail != null) {
      _currentUser = User(
        id: userId,
        name: userName,
        email: userEmail,
        role: userRole ?? 'doctor',
        createdAt: DateTime.now(),
      );
      notifyListeners();
      return true;
    }
    return false;
  }

  Future<bool> login(String email, String password) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // محاكاة تسجيل الدخول
      await Future.delayed(const Duration(seconds: 2));
      
      // في التطبيق الحقيقي، ستقوم بالتحقق من البيانات مع الخادم
      if (email == 'doctor@example.com' && password == '123456') {
        _currentUser = User(
          id: 1,
          name: 'د. أحمد محمد',
          email: email,
          role: 'doctor',
          createdAt: DateTime.now(),
        );

        // حفظ بيانات المستخدم
        final prefs = await SharedPreferences.getInstance();
        await prefs.setInt('userId', _currentUser!.id!);
        await prefs.setString('userName', _currentUser!.name);
        await prefs.setString('userEmail', _currentUser!.email);
        await prefs.setString('userRole', _currentUser!.role);

        notifyListeners();
        return true;
      } else {
        _error = 'البريد الإلكتروني أو كلمة المرور غير صحيحة';
        notifyListeners();
        return false;
      }
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<bool> register({
    required String name,
    required String email,
    required String password,
    required String role,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // محاكاة التسجيل
      await Future.delayed(const Duration(seconds: 2));
      
      _currentUser = User(
        id: DateTime.now().millisecondsSinceEpoch,
        name: name,
        email: email,
        role: role,
        createdAt: DateTime.now(),
      );

      // حفظ بيانات المستخدم
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('userId', _currentUser!.id!);
      await prefs.setString('userName', _currentUser!.name);
      await prefs.setString('userEmail', _currentUser!.email);
      await prefs.setString('userRole', _currentUser!.role);

      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> logout() async {
    _currentUser = null;
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('userId');
    await prefs.remove('userName');
    await prefs.remove('userEmail');
    await prefs.remove('userRole');
    
    notifyListeners();
  }

  Future<bool> updateProfile({
    required String name,
    required String email,
  }) async {
    if (_currentUser == null) return false;

    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      await Future.delayed(const Duration(seconds: 1));
      
      _currentUser = _currentUser!.copyWith(
        name: name,
        email: email,
      );

      // تحديث البيانات المحفوظة
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('userName', name);
      await prefs.setString('userEmail', email);

      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }
}