import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/patient.dart';
import '../models/diagnosis.dart';
import '../models/user.dart';

class DatabaseService {
  static final DatabaseService instance = DatabaseService._init();
  static Database? _database;

  DatabaseService._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('skin_disease.db');
    return _database!;
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);

    return await openDatabase(
      path,
      version: 1,
      onCreate: _createDB,
    );
  }

  Future<void> _createDB(Database db, int version) async {
    // جدول المستخدمين
    await db.execute('''
      CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL,
        profileImage TEXT,
        phone TEXT,
        specialization TEXT,
        licenseNumber TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT
      )
    ''');

    // جدول المرضى
    await db.execute('''
      CREATE TABLE patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT,
        nationalId TEXT NOT NULL UNIQUE,
        dateOfBirth TEXT NOT NULL,
        gender TEXT NOT NULL,
        address TEXT NOT NULL,
        medicalHistory TEXT,
        allergies TEXT,
        currentMedications TEXT,
        emergencyContact TEXT,
        emergencyPhone TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT
      )
    ''');

    // جدول التشخيصات
    await db.execute('''
      CREATE TABLE diagnoses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patientId INTEGER,
        imagePath TEXT NOT NULL,
        predictedClass TEXT NOT NULL,
        className TEXT NOT NULL,
        confidence REAL NOT NULL,
        allProbabilities TEXT NOT NULL,
        notes TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT,
        FOREIGN KEY (patientId) REFERENCES patients (id) ON DELETE CASCADE
      )
    ''');

    // إنشاء فهارس لتحسين الأداء
    await db.execute('CREATE INDEX idx_patients_name ON patients(name)');
    await db.execute('CREATE INDEX idx_patients_phone ON patients(phone)');
    await db.execute('CREATE INDEX idx_diagnoses_patient ON diagnoses(patientId)');
    await db.execute('CREATE INDEX idx_diagnoses_date ON diagnoses(createdAt)');
  }

  // عمليات المرضى
  Future<int> insertPatient(Patient patient) async {
    final db = await instance.database;
    return await db.insert('patients', patient.toMap());
  }

  Future<List<Patient>> getAllPatients() async {
    final db = await instance.database;
    final result = await db.query(
      'patients',
      orderBy: 'createdAt DESC',
    );
    return result.map((map) => Patient.fromMap(map)).toList();
  }

  Future<Patient?> getPatient(int id) async {
    final db = await instance.database;
    final result = await db.query(
      'patients',
      where: 'id = ?',
      whereArgs: [id],
    );
    
    if (result.isNotEmpty) {
      return Patient.fromMap(result.first);
    }
    return null;
  }

  Future<int> updatePatient(Patient patient) async {
    final db = await instance.database;
    return await db.update(
      'patients',
      patient.copyWith(updatedAt: DateTime.now()).toMap(),
      where: 'id = ?',
      whereArgs: [patient.id],
    );
  }

  Future<int> deletePatient(int id) async {
    final db = await instance.database;
    return await db.delete(
      'patients',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<List<Patient>> searchPatients(String query) async {
    final db = await instance.database;
    final result = await db.query(
      'patients',
      where: 'name LIKE ? OR phone LIKE ? OR nationalId LIKE ?',
      whereArgs: ['%$query%', '%$query%', '%$query%'],
      orderBy: 'name ASC',
    );
    return result.map((map) => Patient.fromMap(map)).toList();
  }

  // عمليات التشخيصات
  Future<int> insertDiagnosis(Diagnosis diagnosis) async {
    final db = await instance.database;
    return await db.insert('diagnoses', diagnosis.toMap());
  }

  Future<List<Diagnosis>> getRecentDiagnoses(int limit) async {
    final db = await instance.database;
    final result = await db.query(
      'diagnoses',
      orderBy: 'createdAt DESC',
      limit: limit,
    );
    return result.map((map) => Diagnosis.fromMap(map)).toList();
  }

  Future<List<Diagnosis>> getPatientDiagnoses(int patientId) async {
    final db = await instance.database;
    final result = await db.query(
      'diagnoses',
      where: 'patientId = ?',
      whereArgs: [patientId],
      orderBy: 'createdAt DESC',
    );
    return result.map((map) => Diagnosis.fromMap(map)).toList();
  }

  Future<Diagnosis?> getDiagnosis(int id) async {
    final db = await instance.database;
    final result = await db.query(
      'diagnoses',
      where: 'id = ?',
      whereArgs: [id],
    );
    
    if (result.isNotEmpty) {
      return Diagnosis.fromMap(result.first);
    }
    return null;
  }

  Future<int> updateDiagnosisNotes(int id, String notes) async {
    final db = await instance.database;
    return await db.update(
      'diagnoses',
      {
        'notes': notes,
        'updatedAt': DateTime.now().toIso8601String(),
      },
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> deleteDiagnosis(int id) async {
    final db = await instance.database;
    return await db.delete(
      'diagnoses',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // إحصائيات
  Future<Map<String, int>> getDiagnosisStatistics() async {
    final db = await instance.database;
    final result = await db.rawQuery('''
      SELECT className, COUNT(*) as count
      FROM diagnoses
      GROUP BY className
      ORDER BY count DESC
    ''');
    
    final stats = <String, int>{};
    for (final row in result) {
      stats[row['className'] as String] = row['count'] as int;
    }
    return stats;
  }

  Future<int> getTotalPatients() async {
    final db = await instance.database;
    final result = await db.rawQuery('SELECT COUNT(*) as count FROM patients');
    return result.first['count'] as int;
  }

  Future<int> getTotalDiagnoses() async {
    final db = await instance.database;
    final result = await db.rawQuery('SELECT COUNT(*) as count FROM diagnoses');
    return result.first['count'] as int;
  }

  Future<int> getTodayDiagnoses() async {
    final db = await instance.database;
    final today = DateTime.now();
    final startOfDay = DateTime(today.year, today.month, today.day);
    final endOfDay = startOfDay.add(const Duration(days: 1));
    
    final result = await db.rawQuery('''
      SELECT COUNT(*) as count FROM diagnoses
      WHERE createdAt >= ? AND createdAt < ?
    ''', [startOfDay.toIso8601String(), endOfDay.toIso8601String()]);
    
    return result.first['count'] as int;
  }

  // إغلاق قاعدة البيانات
  Future<void> close() async {
    final db = await instance.database;
    await db.close();
  }
}