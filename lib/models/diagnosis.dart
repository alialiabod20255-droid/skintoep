class Diagnosis {
  final int? id;
  final int? patientId;
  final String imagePath;
  final String predictedClass;
  final String className;
  final double confidence;
  final Map<String, double> allProbabilities;
  final String notes;
  final DateTime createdAt;
  final DateTime? updatedAt;

  Diagnosis({
    this.id,
    this.patientId,
    required this.imagePath,
    required this.predictedClass,
    required this.className,
    required this.confidence,
    required this.allProbabilities,
    required this.notes,
    required this.createdAt,
    this.updatedAt,
  });

  Diagnosis copyWith({
    int? id,
    int? patientId,
    String? imagePath,
    String? predictedClass,
    String? className,
    double? confidence,
    Map<String, double>? allProbabilities,
    String? notes,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return Diagnosis(
      id: id ?? this.id,
      patientId: patientId ?? this.patientId,
      imagePath: imagePath ?? this.imagePath,
      predictedClass: predictedClass ?? this.predictedClass,
      className: className ?? this.className,
      confidence: confidence ?? this.confidence,
      allProbabilities: allProbabilities ?? this.allProbabilities,
      notes: notes ?? this.notes,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'patientId': patientId,
      'imagePath': imagePath,
      'predictedClass': predictedClass,
      'className': className,
      'confidence': confidence,
      'allProbabilities': _encodeProbabilities(allProbabilities),
      'notes': notes,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
    };
  }

  factory Diagnosis.fromMap(Map<String, dynamic> map) {
    return Diagnosis(
      id: map['id']?.toInt(),
      patientId: map['patientId']?.toInt(),
      imagePath: map['imagePath'] ?? '',
      predictedClass: map['predictedClass'] ?? '',
      className: map['className'] ?? '',
      confidence: map['confidence']?.toDouble() ?? 0.0,
      allProbabilities: _decodeProbabilities(map['allProbabilities'] ?? '{}'),
      notes: map['notes'] ?? '',
      createdAt: DateTime.parse(map['createdAt']),
      updatedAt: map['updatedAt'] != null ? DateTime.parse(map['updatedAt']) : null,
    );
  }

  static String _encodeProbabilities(Map<String, double> probabilities) {
    final buffer = StringBuffer('{');
    final entries = probabilities.entries.toList();
    for (int i = 0; i < entries.length; i++) {
      if (i > 0) buffer.write(',');
      buffer.write('"${entries[i].key}":${entries[i].value}');
    }
    buffer.write('}');
    return buffer.toString();
  }

  static Map<String, double> _decodeProbabilities(String encoded) {
    try {
      // تحليل بسيط للـ JSON
      final result = <String, double>{};
      final content = encoded.substring(1, encoded.length - 1); // إزالة الأقواس
      if (content.isEmpty) return result;
      
      final pairs = content.split(',');
      for (final pair in pairs) {
        final parts = pair.split(':');
        if (parts.length == 2) {
          final key = parts[0].replaceAll('"', '').trim();
          final value = double.tryParse(parts[1].trim()) ?? 0.0;
          result[key] = value;
        }
      }
      return result;
    } catch (e) {
      return <String, double>{};
    }
  }

  String get confidencePercentage => '${(confidence * 100).toInt()}%';
  
  String get riskLevel {
    if (confidence >= 0.8) return 'عالي';
    if (confidence >= 0.6) return 'متوسط';
    return 'منخفض';
  }

  Color get riskColor {
    if (confidence >= 0.8) return const Color(0xFFE74C3C);
    if (confidence >= 0.6) return const Color(0xFFF39C12);
    return const Color(0xFF27AE60);
  }

  List<MapEntry<String, double>> get topProbabilities {
    final entries = allProbabilities.entries.toList();
    entries.sort((a, b) => b.value.compareTo(a.value));
    return entries.take(3).toList();
  }

  @override
  String toString() {
    return 'Diagnosis(id: $id, className: $className, confidence: $confidence)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Diagnosis && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;
}