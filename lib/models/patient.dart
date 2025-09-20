class Patient {
  final int? id;
  final String name;
  final String phone;
  final String email;
  final String nationalId;
  final DateTime dateOfBirth;
  final String gender;
  final String address;
  final String medicalHistory;
  final String allergies;
  final String currentMedications;
  final String emergencyContact;
  final String emergencyPhone;
  final DateTime createdAt;
  final DateTime? updatedAt;

  Patient({
    this.id,
    required this.name,
    required this.phone,
    required this.email,
    required this.nationalId,
    required this.dateOfBirth,
    required this.gender,
    required this.address,
    required this.medicalHistory,
    required this.allergies,
    required this.currentMedications,
    required this.emergencyContact,
    required this.emergencyPhone,
    required this.createdAt,
    this.updatedAt,
  });

  Patient copyWith({
    int? id,
    String? name,
    String? phone,
    String? email,
    String? nationalId,
    DateTime? dateOfBirth,
    String? gender,
    String? address,
    String? medicalHistory,
    String? allergies,
    String? currentMedications,
    String? emergencyContact,
    String? emergencyPhone,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return Patient(
      id: id ?? this.id,
      name: name ?? this.name,
      phone: phone ?? this.phone,
      email: email ?? this.email,
      nationalId: nationalId ?? this.nationalId,
      dateOfBirth: dateOfBirth ?? this.dateOfBirth,
      gender: gender ?? this.gender,
      address: address ?? this.address,
      medicalHistory: medicalHistory ?? this.medicalHistory,
      allergies: allergies ?? this.allergies,
      currentMedications: currentMedications ?? this.currentMedications,
      emergencyContact: emergencyContact ?? this.emergencyContact,
      emergencyPhone: emergencyPhone ?? this.emergencyPhone,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'phone': phone,
      'email': email,
      'nationalId': nationalId,
      'dateOfBirth': dateOfBirth.toIso8601String(),
      'gender': gender,
      'address': address,
      'medicalHistory': medicalHistory,
      'allergies': allergies,
      'currentMedications': currentMedications,
      'emergencyContact': emergencyContact,
      'emergencyPhone': emergencyPhone,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
    };
  }

  factory Patient.fromMap(Map<String, dynamic> map) {
    return Patient(
      id: map['id']?.toInt(),
      name: map['name'] ?? '',
      phone: map['phone'] ?? '',
      email: map['email'] ?? '',
      nationalId: map['nationalId'] ?? '',
      dateOfBirth: DateTime.parse(map['dateOfBirth']),
      gender: map['gender'] ?? '',
      address: map['address'] ?? '',
      medicalHistory: map['medicalHistory'] ?? '',
      allergies: map['allergies'] ?? '',
      currentMedications: map['currentMedications'] ?? '',
      emergencyContact: map['emergencyContact'] ?? '',
      emergencyPhone: map['emergencyPhone'] ?? '',
      createdAt: DateTime.parse(map['createdAt']),
      updatedAt: map['updatedAt'] != null ? DateTime.parse(map['updatedAt']) : null,
    );
  }

  int get age {
    final now = DateTime.now();
    int age = now.year - dateOfBirth.year;
    if (now.month < dateOfBirth.month ||
        (now.month == dateOfBirth.month && now.day < dateOfBirth.day)) {
      age--;
    }
    return age;
  }

  String get displayName => name;
  String get displayPhone => phone;
  String get displayEmail => email.isNotEmpty ? email : 'غير محدد';
  String get displayGender => gender == 'male' ? 'ذكر' : 'أنثى';
  
  @override
  String toString() {
    return 'Patient(id: $id, name: $name, phone: $phone, email: $email)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Patient && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;
}